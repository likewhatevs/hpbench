#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <limits.h>
#include <string.h>
#include <ctype.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <linux/mempolicy.h>

#include <linux/mempolicy.h>

/**
 * @file
 * @brief huge page benchmark
 *
 * This benchmark aims to hammer tlb in various real-world-like
 * ways to help quantify the trade-off's of using various page
 * sizes.
 * This populates some memory (executable and not) and hammers 
 * itlb and dtlb while measuring things.
 */

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1MB
#define MAP_HUGE_1MB (20 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif
/**
 * ARRAY_SIZE - count the number of elements in a static array.
 */
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/** Hard cap on threads to prevent runaway configuration. */
static const unsigned MAX_THREADS = 128;

#define DEFAULT_TARGET_RUNTIME_SEC 60.0
#define DEFAULT_REGULAR_BYTES (256UL * 1024 * 1024) /* 256 MiB */
#define DEFAULT_ITERATIONS 50ULL

/**
 * data_target_bytes - derive the total bytes each data benchmark should move.
 *
 * By default the benchmark sweeps three times the system's physical memory to
 * amortise transient effects such as page faults and cache warm-up.  When the
 * kernel withholds physical memory information, a conservative 100 GiB value
 * is used instead.  The result is cached because the expensive sysconf calls
 * only need to run once per process lifetime.
 */
static uint64_t data_target_bytes(void)
{
	static uint64_t cached = 0;
	static bool initialised = false;
	if (initialised) {
		return cached;
	}

	const __uint128_t fallback =
		(__uint128_t)100ULL * 1024ULL * 1024ULL * 1024ULL;
	__uint128_t total = fallback;

	long pages = sysconf(_SC_PHYS_PAGES);
	long page_bytes = sysconf(_SC_PAGESIZE);
	if (pages > 0 && page_bytes > 0) {
		total = (__uint128_t)pages * (uint64_t)page_bytes * 3u;
	}

	if (total == 0) {
		cached = (uint64_t)fallback;
	} else if (total > UINT64_MAX) {
		cached = UINT64_MAX;
	} else {
		cached = (uint64_t)total;
	}

	if (cached == 0) {
		cached = (uint64_t)fallback;
	}

	initialised = true;
	return cached;
}

/**
 * struct bench_config - Process-wide benchmark configuration bundle.
 *
 * The runtime copies user-specified CLI options and derived defaults into this
 * structure once and passes the resulting view everywhere else.  Inline field
 * comments describe how each option influences the individual benchmarks.
 */
struct bench_config {
	unsigned threads; /**< Worker thread count. */
	double target_runtime_sec; /**< Target duration for calibrations and runs. */
	size_t regular_bytes; /**< 4K memory benchmark working-set bytes. */
	uint64_t iterations; /**< Fixed per-thread iteration count. */
	bool disable_perf; /**< Skip perf_event collection when true. */
	bool skip_code; /**< Skip instruction-path benchmarks. */
	bool skip_data; /**< Skip data-path benchmarks. */
	bool skip_code_4k; /**< Skip the 4K-page code benchmark. */
	bool skip_code_huge; /**< Skip the huge-page code benchmark. */
	bool skip_data_4k; /**< Skip the 4K-page data benchmark. */
	bool skip_data_huge; /**< Skip the huge-page data benchmark. */
};

/**
 * Global bookkeeping mirrors common kernel-style benchmarks: a config struct
 * alongside simple globals for perf state and the result sink.
 */
static struct bench_config g_cfg;
/** Flag noting insufficient permissions for perf counting. */
static bool g_perf_permission_error = false;
/** Global reduction sink to keep the compiler from optimising away work. */
static uint64_t g_sink_u64 = 0;

/**
 * struct perf_counter - Perf event metadata and latched readings.
 *
 * Instances of this struct wrap the kernel event descriptor alongside the
 * processed readings so that benchmark code can reason about unavailable
 * counters and scaled multiplexed values in a uniform manner.
 */
struct perf_counter {
	const char *name; /**< Counter label for reports. */
	int fd; /**< Open perf_event file descriptor or -1. */
	bool available; /**< True when the counter is active/usable. */
	uint32_t type; /**< perf_event_attr::type selector. */
	uint64_t config; /**< perf_event_attr::config payload. */
	uint64_t value; /**< Latched counter reading. */
	uint64_t time_enabled; /**< Total time counter was enabled (ns). */
	uint64_t time_running; /**< Total time counter was actively running (ns). */
};

/**
 * struct tlb_stats - Aggregated TLB counter bookkeeping.
 *
 * This helper struct accumulates the raw numbers collected from perf so that
 * each benchmark can print hits, misses, and derived hit rates in a uniform
 * format.
 */
struct tlb_stats {
	bool has_access; /**< True when the access counter is available. */
	bool has_miss; /**< True when the miss counter is available. */
	bool has_hit; /**< True when the hit counter is available. */
	uint64_t accesses; /**< Total TLB accesses recorded. */
	uint64_t hits; /**< Total TLB hits recorded. */
	uint64_t misses; /**< Total TLB misses recorded. */
};

/**
 * struct thread_start_sync - Rendezvous primitive for worker launch.
 *
 * @barrier: POSIX barrier that forces all workers (and the coordinator) to sit
 *           in a holding pen until the benchmark is ready to begin.
 * @go:      Atomic flag flipped by the coordinator to release every waiting
 *           worker once perf counters have been armed and timestamps captured.
 */
struct thread_start_sync {
	pthread_barrier_t barrier; /**< Rendezvous point for all threads. */
	atomic_bool go; /**< Release flag toggled by the controller thread. */
};

static void die_message(const char *msg);
static unsigned autodetect_threads(void);

/**
 * usage - print CLI synopsis using a compact kernel-benchmark style.
 * @prog: Program name used in the usage banner.
 */
static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s [options]\n"
		"  -t, --threads <n>    Number of worker threads\n"
		"  -r, --runtime <sec>  Target runtime per benchmark (default %.2f)\n"
		"  -s, --size <bytes>   Working-set size for 4K data test\n"
		"  -i, --iterations <n> Per-thread iteration count (default %" PRIu64
		")\n"
		"  -P, --no-perf        Disable internal perf counters (validation only)\n"
		"      --skip-code      Skip instruction-path benchmarks\n"
		"      --skip-data      Skip data-path benchmarks\n"
		"      --skip-code-4k      Skip code benchmark on 4K pages\n"
		"      --skip-code-huge     Skip code benchmark on huge pages\n"
		"      --skip-data-4k      Skip data benchmark on 4K pages\n"
		"      --skip-data-huge     Skip data benchmark on huge pages\n"
		"  -h, --help           Show this help\n",
		prog, DEFAULT_TARGET_RUNTIME_SEC, (uint64_t)DEFAULT_ITERATIONS);
}

/**
 * config_init - seed the configuration structure with defaults.
 * @cfg: Configuration object to initialise.
 */
static void config_init(struct bench_config *cfg)
{
	cfg->threads = 0;
	cfg->target_runtime_sec = DEFAULT_TARGET_RUNTIME_SEC;
	cfg->regular_bytes = DEFAULT_REGULAR_BYTES;
	cfg->iterations = DEFAULT_ITERATIONS;
	cfg->disable_perf = false;
	cfg->skip_code = false;
	cfg->skip_data = false;
	cfg->skip_code_4k = false;
	cfg->skip_code_huge = false;
	cfg->skip_data_4k = false;
	cfg->skip_data_huge = false;
}

/**
 * parse_args - kernel-benchmark-style option parsing using getopt_long.
 * @argc: Argument count from main().
 * @argv: Argument vector from main().
 * @cfg: Configuration object to populate with CLI overrides.
 */
static void parse_args(int argc, char **argv, struct bench_config *cfg)
{
	enum {
		OPT_SKIP_CODE = 1000,
		OPT_SKIP_DATA,
		OPT_SKIP_CODE_4K,
		OPT_SKIP_CODE_HUGE,
		OPT_SKIP_DATA_4K,
		OPT_SKIP_DATA_HUGE,
	};

	static const struct option long_opts[] = {
		{ "threads", required_argument, NULL, 't' },
		{ "runtime", required_argument, NULL, 'r' },
		{ "size", required_argument, NULL, 's' },
		{ "iterations", required_argument, NULL, 'i' },
		{ "no-perf", no_argument, NULL, 'P' },
		{ "skip-code", no_argument, NULL, OPT_SKIP_CODE },
		{ "skip-data", no_argument, NULL, OPT_SKIP_DATA },
		{ "skip-code-4k", no_argument, NULL, OPT_SKIP_CODE_4K },
		{ "skip-code-regular", no_argument, NULL, OPT_SKIP_CODE_4K },
		{ "skip-code-huge", no_argument, NULL, OPT_SKIP_CODE_HUGE },
		{ "skip-data-4k", no_argument, NULL, OPT_SKIP_DATA_4K },
		{ "skip-data-regular", no_argument, NULL, OPT_SKIP_DATA_4K },
		{ "skip-data-huge", no_argument, NULL, OPT_SKIP_DATA_HUGE },
		{ "help", no_argument, NULL, 'h' },
		{ NULL, 0, NULL, 0 },
	};

	int opt;

	while ((opt = getopt_long(argc, argv, "t:r:s:i:Ph", long_opts, NULL)) !=
	       -1) {
		switch (opt) {
		case 't': {
			char *end = NULL;
			unsigned long val = strtoul(optarg, &end, 10);
			if (!optarg[0] || !end || *end != '\0' || val == 0) {
				usage(argv[0]);
				die_message("Invalid --threads value");
			}
			cfg->threads = (unsigned)val;
			break;
		}
		case 'r': {
			char *end = NULL;
			double val = strtod(optarg, &end);
			if (!optarg[0] || !end || *end != '\0' || val <= 0.0) {
				usage(argv[0]);
				die_message("Invalid --runtime value");
			}
			cfg->target_runtime_sec = val;
			break;
		}
		case 's': {
			char *end = NULL;
			unsigned long long val = strtoull(optarg, &end, 0);
			if (!optarg[0] || !end || *end != '\0' || val == 0ULL) {
				usage(argv[0]);
				die_message("Invalid --size value");
			}
			cfg->regular_bytes = (size_t)val;
			break;
		}
		case 'i': {
			char *end = NULL;
			unsigned long long val = strtoull(optarg, &end, 10);
			if (!optarg[0] || !end || *end != '\0' || val == 0ULL) {
				usage(argv[0]);
				die_message("Invalid --iterations value");
			}
			cfg->iterations = (uint64_t)val;
			break;
		}
		case 'P':
			cfg->disable_perf = true;
			break;
		case OPT_SKIP_CODE:
			cfg->skip_code = true;
			break;
		case OPT_SKIP_DATA:
			cfg->skip_data = true;
			break;
		case OPT_SKIP_CODE_4K:
			cfg->skip_code_4k = true;
			break;
		case OPT_SKIP_CODE_HUGE:
			cfg->skip_code_huge = true;
			break;
		case OPT_SKIP_DATA_4K:
			cfg->skip_data_4k = true;
			break;
		case OPT_SKIP_DATA_HUGE:
			cfg->skip_data_huge = true;
			break;
		case 'h':
			usage(argv[0]);
			exit(EXIT_SUCCESS);
		default:
			usage(argv[0]);
			die_message("Unknown option");
		}
	}

	if (optind != argc) {
		usage(argv[0]);
		die_message("Unexpected positional arguments");
	}
}

/**
 * normalize_config - finalise configuration after CLI/env processing.
 * @cfg: Configuration object to clamp and top up with defaults.
 */
static void normalize_config(struct bench_config *cfg)
{
	if (cfg->threads == 0) {
		cfg->threads = autodetect_threads();
	}
	if (cfg->threads < 2) {
		cfg->threads = 2;
	}
	if (cfg->threads > MAX_THREADS) {
		cfg->threads = MAX_THREADS;
	}
	if (cfg->target_runtime_sec <= 0.0) {
		cfg->target_runtime_sec = DEFAULT_TARGET_RUNTIME_SEC;
	}
	if (cfg->regular_bytes == 0) {
		cfg->regular_bytes = DEFAULT_REGULAR_BYTES;
	}
	if (cfg->iterations == 0) {
		cfg->iterations = DEFAULT_ITERATIONS;
	}
}

/**
 * flush_tlb - best-effort eviction of architectural TLB entries.
 *
 * The micro-benchmarks are sensitive to residual translations left behind by
 * previous runs.  Touching a large, throw-away mapping forces the hardware to
 * repopulate both instruction and data TLBs before the real workload begins,
 * reducing run-to-run variance in the reported miss counts.
 */
static void flush_tlb(void)
{
	size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
	if (page_size == 0) {
		page_size = 4096;
	}

	const size_t pages =
		16384; /* 64 MiB at 4 KiB pages, plenty to evict TLBs. */
	size_t length = pages * page_size;

	uint8_t *buf = mmap(NULL, length, PROT_READ | PROT_WRITE,
			    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (buf == MAP_FAILED) {
		return;
	}

	for (size_t i = 0; i < pages; ++i) {
		volatile uint8_t *ptr = buf + i * page_size;
		*ptr = (uint8_t)i;
	}

	munmap(buf, length);
}

static uint32_t cpu_pmu_type(void)
{
	static uint32_t type_cache;
	static bool cached = false;
	if (cached) {
		return type_cache;
	}

	const char *path = "/sys/bus/event_source/devices/cpu/type";
	FILE *fp = fopen(path, "r");
	if (!fp) {
		die_message("Unable to open CPU PMU type");
	}

	unsigned long value = 0;
	if (fscanf(fp, "%lu", &value) != 1) {
		fclose(fp);
		die_message("Failed to parse CPU PMU type");
	}
	fclose(fp);

	type_cache = (uint32_t)value;
	cached = true;
	return type_cache;
}

/**
 * trim_whitespace - Remove leading/trailing ASCII whitespace in-place.
 * @str: Mutable string to clean up.
 *
 * Return: Pointer to the first non-space character within @str (which may be
 *         the original pointer if no trimming was required).
 */
static char *trim_whitespace(char *str)
{
	while (*str && isspace((unsigned char)*str)) {
		++str;
	}
	if (*str == '\0') {
		return str;
	}
	char *end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end)) {
		*end-- = '\0';
	}
	return str;
}

/**
 * read_sysfs_event_spec - Locate an event encoding via the sysfs perf tree.
 * @event_name: Kernel event alias (e.g. "bp_l1_tlb_fetch_hit").
 * @buf: Output buffer for the raw config string.
 * @len: Size of @buf in bytes.
 *
 * Return: true on success, false if the event entry could not be read.
 */
static bool read_sysfs_event_spec(const char *event_name, char *buf, size_t len)
{
	char path[PATH_MAX];
	int written = snprintf(path, sizeof(path),
			       "/sys/bus/event_source/devices/cpu/events/%s",
			       event_name);
	if (written <= 0 || (size_t)written >= sizeof(path)) {
		return false;
	}

	FILE *fp = fopen(path, "r");
	if (!fp) {
		return false;
	}

	bool ok = fgets(buf, (int)len, fp) != NULL;
	fclose(fp);
	if (!ok) {
		return false;
	}

	char *newline = strchr(buf, '\n');
	if (newline) {
		*newline = '\0';
	}

	char *trimmed = trim_whitespace(buf);
	if (trimmed != buf) {
		memmove(buf, trimmed, strlen(trimmed) + 1);
	}

	return buf[0] != '\0';
}

/**
 * fetch_event_spec_via_perf - Fallback lookup using `perf list --details`.
 * @event_name: Kernel event alias.
 * @buf: Output buffer for the raw config string.
 * @len: Size of @buf in bytes.
 *
 * Return: true when parsing succeeded.
 */
static bool fetch_event_spec_via_perf(const char *event_name, char *buf,
				      size_t len)
{
	char cmd[512];
	int written = snprintf(cmd, sizeof(cmd),
			       "perf list --details %s 2>/dev/null",
			       event_name);
	if (written <= 0 || (size_t)written >= sizeof(cmd)) {
		return false;
	}

	FILE *pipe = popen(cmd, "r");
	if (!pipe) {
		return false;
	}

	char line[512];
	bool found = false;
	while (fgets(line, sizeof(line), pipe)) {
		char *cpu = strstr(line, "cpu/");
		if (!cpu) {
			continue;
		}
		char *spec = cpu + 4;
		char *end = strchr(spec, '/');
		if (!end) {
			continue;
		}
		size_t spec_len = (size_t)(end - spec);
		if (spec_len == 0) {
			continue;
		}
		if (spec_len >= len) {
			spec_len = len - 1;
		}
		memcpy(buf, spec, spec_len);
		buf[spec_len] = '\0';
		found = true;
		break;
	}

	pclose(pipe);
	return found;
}

/**
 * resolve_cpu_event_spec - Locate an architectural event description.
 * @event_name: Kernel event alias.
 * @buf: Output buffer for the raw config string.
 * @len: Size of @buf in bytes.
 *
 * Return: true when the spec could be located via sysfs or perf.
 */
static bool resolve_cpu_event_spec(const char *event_name, char *buf,
				   size_t len)
{
	if (read_sysfs_event_spec(event_name, buf, len)) {
		return true;
	}
	return fetch_event_spec_via_perf(event_name, buf, len);
}

/**
 * parse_cpu_event_config - Convert "event=..." specs into perf configs.
 * @spec: Raw spec string (e.g. "event=0x94,umask=0xff").
 * @config_out: Location to store the encoded config value on success.
 *
 * Return: true when parsing succeeded.
 */
static bool parse_cpu_event_config(const char *spec, uint64_t *config_out)
{
	if (!spec || !config_out) {
		return false;
	}

	char tmp[256];
	strncpy(tmp, spec, sizeof(tmp));
	tmp[sizeof(tmp) - 1] = '\0';

	uint64_t event = 0;
	uint64_t umask = 0;
	uint64_t cmask = 0;
	bool edge = false;
	bool inv = false;
	bool have_event = false;

	char *saveptr = NULL;
	char *token = strtok_r(tmp, ",", &saveptr);
	while (token) {
		char *part = trim_whitespace(token);
		if (strncmp(part, "event=", 6) == 0) {
			event = strtoull(part + 6, NULL, 0);
			have_event = true;
		} else if (strncmp(part, "umask=", 6) == 0) {
			umask = strtoull(part + 6, NULL, 0);
		} else if (strncmp(part, "cmask=", 6) == 0) {
			cmask = strtoull(part + 6, NULL, 0);
		} else if (strncmp(part, "edge", 4) == 0) {
			if (part[4] == '=') {
				edge = strtoull(part + 5, NULL, 0) != 0;
			} else {
				edge = true;
			}
		} else if (strncmp(part, "inv", 3) == 0) {
			if (part[3] == '=') {
				inv = strtoull(part + 4, NULL, 0) != 0;
			} else {
				inv = true;
			}
		}
		token = strtok_r(NULL, ",", &saveptr);
	}

	if (!have_event) {
		return false;
	}

	uint64_t config = event & 0xffULL;
	config |= (umask & 0xffULL) << 8;
	if (edge) {
		config |= 1ULL << 18;
	}
	if (inv) {
		config |= 1ULL << 23;
	}
	if (cmask) {
		config |= (cmask & 0xffULL) << 24;
	}

	*config_out = config;
	return true;
}

/**
 * sys_perf_event_open - thin syscall wrapper so we do not drag in libc helpers.
 * @attr: Event attributes to register.
 * @pid: Target pid (0 for self).
 * @cpu: Target CPU (-1 for any).
 * @group_fd: Group file descriptor (-1 for none).
 * @flags: Additional flags for the syscall.
 *
 * Return: File descriptor on success or -1 on failure with errno set.
 */
static long sys_perf_event_open(struct perf_event_attr *attr, pid_t pid,
				int cpu, int group_fd, unsigned long flags)
{
	return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/**
 * die_errno - bail out with errno preserved.
 * @msg: Context string to print alongside strerror.
 */
static void die_errno(const char *msg)
{
	perror(msg);
	exit(EXIT_FAILURE);
}

/**
 * die_message - helper for hard logic failures.
 * @msg: Message to emit before exiting.
 */
static void die_message(const char *msg)
{
	fputs(msg, stderr);
	fputc('\n', stderr);
	exit(EXIT_FAILURE);
}

/**
 * cpu_relax - architecture hint for busy-waits.
 */
static inline void cpu_relax(void)
{
	__asm__ __volatile__("pause" ::: "memory");
}

/**
 * thread_start_sync_init - prepare the barrier + flag pair used to line up workers.
 * @sync: Synchronisation primitive to initialise.
 * @thread_count: Number of worker threads participating.
 */
static void thread_start_sync_init(struct thread_start_sync *sync,
				   unsigned thread_count)
{
	if (pthread_barrier_init(&sync->barrier, NULL, thread_count + 1) != 0) {
		die_errno("pthread_barrier_init");
	}
	atomic_init(&sync->go, false);
}

/**
 * thread_start_sync_destroy - release resources paired with init.
 * @sync: Synchronisation primitive to destroy.
 */
static void thread_start_sync_destroy(struct thread_start_sync *sync)
{
	if (pthread_barrier_destroy(&sync->barrier) != 0) {
		die_errno("pthread_barrier_destroy");
	}
}

/**
 * wait_for_start - common entry point for worker threads.
 * @sync: Synchronisation primitive to wait on.
 */
static void wait_for_start(struct thread_start_sync *sync)
{
	pthread_barrier_wait(&sync->barrier);
	while (!atomic_load_explicit(&sync->go, memory_order_acquire)) {
		cpu_relax();
	}
}

/**
 * init_counter - open one perf event and record whether it succeeded.
 * @ctr: Counter wrapper to populate.
 * @name: Human-readable identifier for logs.
 * @type: perf_event type selector.
 * @config: perf_event config selector.
 */
static void init_counter(struct perf_counter *ctr, const char *name,
			 uint32_t type, uint64_t config)
{
	memset(ctr, 0, sizeof(*ctr));
	ctr->name = name;
	ctr->type = type;
	ctr->config = config;
	ctr->fd = -1;

	struct perf_event_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.type = type;
	attr.size = sizeof(attr);
	attr.config = config;
	attr.disabled = 1;
	attr.exclude_kernel = 1;
	attr.exclude_hv = 1;
	attr.inherit = 1; /* include child threads */
	attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
			   PERF_FORMAT_TOTAL_TIME_RUNNING;

	int fd = sys_perf_event_open(&attr, 0, -1, -1, 0);
	if (fd == -1) {
		ctr->available = false;
		if (errno == EACCES || errno == EPERM) {
			g_perf_permission_error = true;
		}
		return;
	}

	ctr->fd = fd;
	ctr->available = true;
}

static void init_named_cpu_counter(struct perf_counter *ctr, const char *label,
				   const char *event_name)
{
	char spec[256];
	if (!resolve_cpu_event_spec(event_name, spec, sizeof(spec))) {
		fprintf(stderr, "Unable to resolve event descriptor for '%s'\n",
			event_name);
		die_message("Missing perf event spec");
	}

	uint64_t config = 0;
	if (!parse_cpu_event_config(spec, &config)) {
		fprintf(stderr,
			"Unable to parse event descriptor '%s' for '%s'\n",
			spec, event_name);
		die_message("Failed to parse perf event spec");
	}

	init_counter(ctr, label, cpu_pmu_type(), config);
	if (!ctr->available) {
		fprintf(stderr, "Required event '%s' unavailable (config %s)\n",
			event_name, spec);
		die_message("perf event open failed");
	}
}

static uint64_t make_cache_config(unsigned cache, unsigned op, unsigned result)
{
	return ((uint64_t)cache) | ((uint64_t)op << 8) |
	       ((uint64_t)result << 16);
}

static void init_cache_counter(struct perf_counter *ctr, const char *name,
			       unsigned cache, unsigned op, unsigned result)
{
	uint64_t config = make_cache_config(cache, op, result);
	init_counter(ctr, name, PERF_TYPE_HW_CACHE, config);
}

/**
 * close_counter - helper to centralise fd teardown.
 * @ctr: Counter wrapper to close.
 */
static void close_counter(struct perf_counter *ctr)
{
	if (ctr->fd != -1) {
		close(ctr->fd);
		ctr->fd = -1;
	}
	ctr->available = false;
}

/**
 * enable_counters - reset+start each active counter.
 * @counters: Array of counter wrappers.
 * @count: Number of entries in @counters.
 */
static void enable_counters(struct perf_counter *counters, size_t count)
{
	for (size_t i = 0; i < count; ++i) {
		struct perf_counter *ctr = &counters[i];
		if (!ctr->available) {
			continue;
		}
		if (ioctl(ctr->fd, PERF_EVENT_IOC_RESET, 0) == -1 ||
		    ioctl(ctr->fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
			ctr->available = false;
			close_counter(ctr);
		}
	}
}

/**
 * disable_counters - stop counters and read back their values.
 * @counters: Array of counter wrappers.
 * @count: Number of entries in @counters.
 */
static void disable_counters(struct perf_counter *counters, size_t count)
{
	for (size_t i = 0; i < count; ++i) {
		struct perf_counter *ctr = &counters[i];
		if (!ctr->available) {
			continue;
		}
		if (ioctl(ctr->fd, PERF_EVENT_IOC_DISABLE, 0) == -1) {
			ctr->available = false;
			close_counter(ctr);
			continue;
		}
		struct {
			uint64_t value;
			uint64_t time_enabled;
			uint64_t time_running;
		} sample;

		ssize_t read_bytes = read(ctr->fd, &sample, sizeof(sample));
		if (read_bytes != (ssize_t)sizeof(sample)) {
			ctr->available = false;
			close_counter(ctr);
			continue;
		}
		ctr->time_enabled = sample.time_enabled;
		ctr->time_running = sample.time_running;

		uint64_t scaled_value = sample.value;
		if (sample.time_running == 0) {
			ctr->available = false;
			close_counter(ctr);
			continue;
		}
		if (sample.time_running < sample.time_enabled) {
			__uint128_t scaled =
				(__uint128_t)sample.value * sample.time_enabled;
			scaled /= sample.time_running;
			if (scaled > UINT64_MAX) {
				scaled_value = UINT64_MAX;
			} else {
				scaled_value = (uint64_t)scaled;
			}
		}
		ctr->value = scaled_value;
	}
}

/**
 * cleanup_counters - blanket close helper used before program exit.
 * @counters: Array of counter wrappers.
 * @count: Number of entries in @counters.
 */
static void cleanup_counters(struct perf_counter *counters, size_t count)
{
	for (size_t i = 0; i < count; ++i) {
		close_counter(&counters[i]);
	}
}

/**
 * safe_add_u64 - Perform saturating addition on 64-bit unsigned integers.
 * @a: First operand.
 * @b: Second operand.
 *
 * Return: a + b unless the sum would overflow, in which case UINT64_MAX.
 */
static uint64_t safe_add_u64(uint64_t a, uint64_t b)
{
	if (UINT64_MAX - a < b) {
		return UINT64_MAX;
	}
	return a + b;
}

/**
 * record_tlb_counter - Incorporate a perf counter into aggregated TLB stats.
 * @stats: Aggregation bucket to update.
 * @name: Name of the perf counter being recorded.
 * @value: Latched counter value.
 */
static void record_tlb_counter(struct tlb_stats *stats, const char *name,
			       uint64_t value)
{
	if (!name) {
		return;
	}

	if (strcmp(name, "bp_l1_tlb_fetch_hit") == 0 ||
	    strcmp(name, "bp_l1_tlb_miss_l2_hit") == 0) {
		stats->has_hit = true;
		stats->hits = safe_add_u64(stats->hits, value);
		return;
	}
	if (strcmp(name, "bp_l1_tlb_miss_l2_tlb_miss") == 0) {
		stats->has_miss = true;
		stats->misses = safe_add_u64(stats->misses, value);
		return;
	}
	if (strcmp(name, "dTLB-loads") == 0) {
		stats->has_access = true;
		stats->accesses = value;
		return;
	}
	if (strcmp(name, "dTLB-load-misses") == 0) {
		stats->has_miss = true;
		stats->misses = safe_add_u64(stats->misses, value);
		return;
	}
}

/**
 * is_itlb_label - Identify whether a counter label belongs to ITLB metrics.
 * @name: Counter name string.
 */
static bool is_itlb_label(const char *name)
{
	return name && (strcmp(name, "bp_l1_tlb_fetch_hit") == 0 ||
			strcmp(name, "bp_l1_tlb_miss_l2_hit") == 0 ||
			strcmp(name, "bp_l1_tlb_miss_l2_tlb_miss") == 0);
}

/**
 * is_dtlb_label - Identify whether a counter label belongs to DTLB metrics.
 * @name: Counter name string.
 */
static bool is_dtlb_label(const char *name)
{
	return name && (strcmp(name, "dTLB-loads") == 0 ||
			strcmp(name, "dTLB-load-misses") == 0);
}

/**
 * finalize_tlb_stats - Derive missing TLB fields from the available counters.
 * @stats: Aggregated stats structure to normalise.
 */
static void finalize_tlb_stats(struct tlb_stats *stats)
{
	if (!stats->has_access && stats->has_hit && stats->has_miss) {
		stats->accesses = safe_add_u64(stats->hits, stats->misses);
		stats->has_access = true;
	}
	if (stats->has_access && stats->has_miss) {
		if (stats->accesses >= stats->misses) {
			stats->hits = stats->accesses - stats->misses;
		} else {
			stats->hits = 0;
		}
		stats->has_hit = true;
	}
	if (!stats->has_miss && stats->has_access && !stats->has_hit) {
		stats->hits = stats->accesses;
		stats->has_hit = true;
	}
	if (!stats->has_miss && stats->has_access && stats->has_hit) {
		if (stats->accesses >= stats->hits) {
			stats->misses = stats->accesses - stats->hits;
		} else {
			stats->misses = 0;
		}
		stats->has_miss = true;
	}
}

/**
 * print_tlb_stats - Emit a formatted TLB summary for the current benchmark.
 * @label: Human-readable TLB label (e.g. "ITLB").
 * @stats: Aggregated stats collected for that label.
 */
static void print_tlb_stats(const char *label, struct tlb_stats stats)
{
	finalize_tlb_stats(&stats);

	double hit_rate = -1.0;
	if (stats.has_access && stats.has_hit) {
		if (stats.accesses > 0) {
			hit_rate = ((double)stats.hits * 100.0) /
				   (double)stats.accesses;
		} else {
			hit_rate = stats.hits > 0 ? 100.0 : 0.0;
		}
	} else if (stats.has_hit && stats.has_miss) {
		uint64_t total = safe_add_u64(stats.hits, stats.misses);
		if (total > 0) {
			hit_rate = ((double)stats.hits * 100.0) / (double)total;
		} else {
			hit_rate = 0.0;
		}
	} else if (stats.has_hit && !stats.has_miss) {
		hit_rate = 100.0;
	}

	if (stats.has_access && stats.has_hit && stats.has_miss) {
		printf("  %s: %" PRIu64 " accesses, %" PRIu64 " hits, %" PRIu64
		       " misses\n",
		       label, stats.accesses, stats.hits, stats.misses);
	} else if (stats.has_hit && stats.has_miss) {
		printf("  %s: %" PRIu64 " hits, %" PRIu64 " misses\n", label,
		       stats.hits, stats.misses);
	} else if (stats.has_miss) {
		printf("  %s: misses only (%" PRIu64 " misses)\n", label,
		       stats.misses);
	} else if (stats.has_access) {
		printf("  %s: accesses only (%" PRIu64 " accesses)\n", label,
		       stats.accesses);
	} else if (stats.has_hit) {
		printf("  %s: hits only (%" PRIu64 " hits)\n", label,
		       stats.hits);
	} else {
		printf("  %s: counters unavailable\n", label);
	}

	if (hit_rate >= 0.0) {
		if (hit_rate > 100.0) {
			hit_rate = 100.0;
		}
		printf("  %s hit rate: %.6f%%\n", label, hit_rate);
	}
}

/**
 * select_data_iterations - compute iteration count to hit the byte target.
 *
 * The data benchmarks process both a source and destination mapping each loop,
 * so the per-iteration cost is twice the map size.  This helper translates the
 * global byte target into a loop count while defending against overflow and
 * zero-sized mappings.
 */
static uint64_t select_data_iterations(size_t map_size)
{
	if (map_size == 0) {
		return 1;
	}

	const unsigned __int128 target_bytes =
		(unsigned __int128)data_target_bytes();
	const unsigned __int128 per_iteration =
		(unsigned __int128)map_size * 2u;

	if (per_iteration == 0) {
		return 1;
	}

	unsigned __int128 iterations = target_bytes / per_iteration;
	if ((target_bytes % per_iteration) != 0) {
		++iterations;
	}
	if (iterations == 0) {
		iterations = 1;
	}
	if (iterations > UINT64_MAX) {
		return UINT64_MAX;
	}
	return (uint64_t)iterations;
}

/**
 * timespec_diff - convert two struct timespec samples into seconds.
 * @start: Earlier timestamp.
 * @end: Later timestamp.
 *
 * Return: Difference in seconds as a double.
 */
static double timespec_diff(const struct timespec *start,
			    const struct timespec *end)
{
	double sec = (double)(end->tv_sec - start->tv_sec);
	double nsec = (double)(end->tv_nsec - start->tv_nsec) * 1e-9;
	return sec + nsec;
}

/**
 * autodetect_threads - fallback when thread count is unspecified.
 *
 * Return: Reasonable worker count based on online CPUs.
 */
static unsigned autodetect_threads(void)
{
	unsigned threads = 0;
	long onln = sysconf(_SC_NPROCESSORS_ONLN);

	if (onln > 0) {
		long scaled = (onln * 3L + 1L) / 2L; /* round up 1.5x */
		if (scaled > LONG_MAX) {
			scaled = LONG_MAX;
		}
		threads = (unsigned)scaled;
	}
	if (threads < 2) {
		threads = 2;
	}
	if (threads > MAX_THREADS) {
		threads = MAX_THREADS;
	}
	return threads;
}

/**
 * bind_memory_to_local_node - prefer the NUMA node of the caller for @addr.
 * @addr: Mapping base address.
 * @length: Mapping span in bytes.
 */
static void bind_memory_to_local_node(void *addr, size_t length)
{
	if (!addr || length == 0) {
		return;
	}

	unsigned cpu = 0;
	unsigned node = 0;
	if (syscall(__NR_getcpu, &cpu, &node, NULL) != 0) {
		return;
	}

	const unsigned bits_per_word =
		(unsigned)(sizeof(unsigned long) * CHAR_BIT);
	const unsigned max_supported_nodes = 256;
	if (bits_per_word == 0 || node >= max_supported_nodes) {
		return;
	}

	unsigned long nodemask[(max_supported_nodes + bits_per_word - 1) /
			       bits_per_word];
	memset(nodemask, 0, sizeof(nodemask));
	const unsigned word = node / bits_per_word;
	const unsigned bit = node % bits_per_word;
	nodemask[word] |= 1UL << bit;

#ifndef MPOL_PREFERRED
#define MPOL_PREFERRED 1
#endif

	unsigned long maxnode = (unsigned long)(node + 1);
	(void)syscall(__NR_mbind, addr, (unsigned long)length, MPOL_PREFERRED,
		      nodemask, maxnode, 0UL);
}

/**
 * map_buffer - mmap wrapper used for both executable and data buffers.
 * @length: Mapping length in bytes.
 * @prot: Protection flags (PROT_*).
 * @extra_flags: Additional MAP_* flags to OR in.
 *
 * Return: Pointer to the mapping or NULL on failure.
 */
static void *map_buffer(size_t length, int prot, int extra_flags)
{
	void *addr = mmap(NULL, length, prot,
			  MAP_PRIVATE | MAP_ANONYMOUS | extra_flags, -1, 0);
	if (addr == MAP_FAILED) {
		return NULL;
	}
	bind_memory_to_local_node(addr, length);
	return addr;
}

/**
 * release_mapping - munmap wrapper with NULL protection.
 * @addr: Mapping address (ignored if NULL).
 * @length: Mapping length.
 */
static void release_mapping(void *addr, size_t length)
{
	if (addr) {
		munmap(addr, length);
	}
}

typedef uint64_t (*jit_kernel_fn)(uint64_t iterations);

/**
 * prepare_jit_kernel - allocate and populate an executable code buffer.
 * @map_size: Size of the executable mapping.
 * @extra_flags: Extra MAP_* flags to request (e.g. MAP_HUGETLB).
 * @mapping_out: Optional return of the raw mapping address.
 *
 * Return: Function pointer to the generated kernel or NULL on failure.
 */
static jit_kernel_fn prepare_jit_kernel(size_t map_size, int extra_flags,
					void **mapping_out)
{
	void *mapping = map_buffer(map_size, PROT_READ | PROT_WRITE | PROT_EXEC,
				   extra_flags);
	if (!mapping) {
		return NULL;
	}
	uint8_t *code = (uint8_t *)mapping;
	size_t offset = 0;

	/* Prologue: seed loop counter, working registers and accumulator. */
	code[offset++] = 0x48;
	code[offset++] = 0x89;
	code[offset++] = 0xF9; /* mov rcx, rdi */
	code[offset++] = 0x48;
	code[offset++] = 0x89;
	code[offset++] = 0xFA; /* mov rdx, rdi */
	code[offset++] = 0x48;
	code[offset++] = 0x31;
	code[offset++] = 0xC0; /* xor rax, rax */
	code[offset++] = 0x48;
	code[offset++] = 0x85;
	code[offset++] = 0xC9; /* test rcx, rcx */

	size_t je_patch = offset;
	code[offset++] = 0x0F;
	code[offset++] = 0x84; /* je done (rel32 placeholder) */
	offset += 4;

	size_t loop_pos = offset;

	size_t call_patch = offset;
	code[offset++] = 0xE8; /* call slot0 */
	offset += 4;

	code[offset++] = 0x48;
	code[offset++] = 0xFF;
	code[offset++] = 0xC9; /* dec rcx */

	size_t jne_patch = offset;
	code[offset++] = 0x0F;
	code[offset++] = 0x85; /* jne loop (rel32 placeholder) */
	offset += 4;

	size_t done_pos = offset;
	code[offset++] = 0xC3; /* ret */

	const size_t slot_size = 32;
	const size_t max_slots = 2097152;

	size_t slots_start = offset;
	size_t slots_space = map_size > slots_start ? map_size - slots_start :
						      0;
	size_t max_possible_slots = slots_space / slot_size;
	if (max_possible_slots == 0) {
		release_mapping(mapping, map_size);
		return NULL;
	}
	size_t slot_count = max_possible_slots;
	if (slot_count > max_slots) {
		slot_count = max_slots;
	}

	size_t stride = (slot_count > 1) ? (slot_count - 1) : 0;

	size_t *order = malloc(slot_count * sizeof(*order));
	if (!order) {
		release_mapping(mapping, map_size);
		return NULL;
	}
	size_t current = 0;
	for (size_t i = 0; i < slot_count; ++i) {
		order[i] = current;
		if (slot_count > 1) {
			current = (current + stride) % slot_count;
		}
	}
	if (slot_count > 1 && current != 0) {
		for (size_t i = 0; i < slot_count; ++i) {
			order[i] = i;
		}
	}

	size_t *next_slot = malloc(slot_count * sizeof(*next_slot));
	if (!next_slot) {
		free(order);
		release_mapping(mapping, map_size);
		return NULL;
	}
	for (size_t i = 0; i < slot_count; ++i) {
		size_t slot_index = order[i];
		next_slot[slot_index] = (i + 1 < slot_count) ? order[i + 1] :
							       slot_count;
	}
	free(order);

	for (size_t i = 0; i < slot_count; ++i) {
		size_t slot_offset = slots_start + i * slot_size;
		uint8_t *slot = code + slot_offset;
		size_t pos = 0;

		slot[pos++] = 0x48;
		slot[pos++] = 0x01;
		slot[pos++] = 0xF8; /* add rax, rdi */

		slot[pos++] = 0x48;
		slot[pos++] = 0x05; /* add rax, imm32 */
		uint32_t add_imm = 0x9E3779B9u + (uint32_t)(i * 0x51EDu);
		memcpy(slot + pos, &add_imm, sizeof(add_imm));
		pos += sizeof(add_imm);

		slot[pos++] = 0x48;
		slot[pos++] = 0xC1;
		slot[pos++] = 0xC0; /* rol rax, imm8 */
		slot[pos++] = (uint8_t)((i & 0x7) + 1);

		slot[pos++] = 0x48;
		slot[pos++] = 0x31;
		slot[pos++] = 0xD0; /* xor rax, rdx */
		slot[pos++] = 0x48;
		slot[pos++] = 0x01;
		slot[pos++] = 0xD0; /* add rax, rdx */

		size_t next_index = next_slot[i];
		if (next_index < slot_count) {
			slot[pos++] = 0xE9; /* jmp next slot */
			int32_t rel =
				(int32_t)((int64_t)(slots_start +
						    next_index * slot_size) -
					  (int64_t)(slot_offset + pos + 4));
			memcpy(slot + pos, &rel, sizeof(rel));
			pos += sizeof(rel);
		} else {
			slot[pos++] = 0xC3; /* ret to caller */
		}

		while (pos < slot_size) {
			slot[pos++] = 0x90; /* nop padding */
		}
	}

	/* Patch branch targets now that layout is final. */
	int32_t call_disp =
		(int32_t)((int64_t)slots_start - (int64_t)(call_patch + 5));
	memcpy(&code[call_patch + 1], &call_disp, sizeof(call_disp));

	int32_t jne_disp =
		(int32_t)((int64_t)loop_pos - (int64_t)(jne_patch + 6));
	memcpy(&code[jne_patch + 2], &jne_disp, sizeof(jne_disp));

	int32_t je_disp =
		(int32_t)((int64_t)done_pos - (int64_t)(je_patch + 6));
	memcpy(&code[je_patch + 2], &je_disp, sizeof(je_disp));

	free(next_slot);

	__builtin___clear_cache((char *)mapping, (char *)mapping + map_size);
	if (mapping_out) {
		*mapping_out = mapping;
	}
	return (jit_kernel_fn)mapping;
}

/**
 * struct code_thread_args - Per-thread state for the instruction benchmark.
 *
 * Holds the JIT entry point, iteration budget, and synchronisation handle used
 * by each worker thread participating in the code-path stress test.
 */
struct code_thread_args {
	jit_kernel_fn fn; /**< Executable kernel to run. */
	uint64_t iterations; /**< Loop iterations assigned to the thread. */
	struct thread_start_sync
		*sync; /**< Start-line synchronisation primitive. */
};

/**
 * code_thread_main - worker loop for instruction-path benchmarking.
 * @arg: Pointer to struct code_thread_args describing the work item.
 *
 * Return: NULL (the pthread exit value is unused).
 */
static void *code_thread_main(void *arg)
{
	struct code_thread_args *ctx = (struct code_thread_args *)arg;
	wait_for_start(ctx->sync);
	uint64_t result = ctx->fn(ctx->iterations);
	__atomic_fetch_xor(&g_sink_u64, result, __ATOMIC_RELAXED);
	return NULL;
}

/**
 * execute_code_kernel_mt - launch worker threads against the same JIT kernel.
 * @fn: Executable kernel to invoke.
 * @iterations: Per-thread iteration budget.
 * @thread_count: Number of worker threads to spawn.
 * @counters: Optional perf counter array.
 * @counter_count: Number of counters in @counters.
 * @collect_perf: Whether to enable and sample the counters.
 *
 * Return: Elapsed time in seconds.
 */
static double execute_code_kernel_mt(jit_kernel_fn fn, uint64_t iterations,
				     unsigned thread_count)
{
	struct thread_start_sync sync;
	thread_start_sync_init(&sync, thread_count);

	pthread_t *threads = calloc(thread_count, sizeof(*threads));
	struct code_thread_args *args = calloc(thread_count, sizeof(*args));
	if (!threads || !args) {
		die_message("Failed to allocate thread bookkeeping");
	}

	for (unsigned i = 0; i < thread_count; ++i) {
		/* Each worker gets the function pointer, iteration budget and sync obj. */
		args[i].fn = fn;
		args[i].iterations = iterations;
		args[i].sync = &sync;
		if (pthread_create(&threads[i], NULL, code_thread_main,
				   &args[i]) != 0) {
			die_errno("pthread_create (code)");
		}
	}

	pthread_barrier_wait(&sync.barrier);

	struct timespec start_ts = { 0 }, end_ts = { 0 };
	clock_gettime(CLOCK_MONOTONIC, &start_ts);
	/* Flipping the go-flag releases every worker more or less simultaneously. */
	atomic_store_explicit(&sync.go, true, memory_order_release);

	for (unsigned i = 0; i < thread_count; ++i) {
		pthread_join(threads[i], NULL);
	}
	clock_gettime(CLOCK_MONOTONIC, &end_ts);

	thread_start_sync_destroy(&sync);
	free(args);
	free(threads);

	return timespec_diff(&start_ts, &end_ts);
}

/**
 * run_code_benchmark - front-end for the instruction benchmark.
 * @label: Benchmark name used in logs.
 * @map_size: Size of the executable mapping to allocate.
 * @extra_flags: Additional MAP_* flags (e.g. MAP_HUGETLB).
 * @counters: Perf counter array to use for measurements.
 * @counter_count: Number of entries in @counters.
 */
static void run_code_benchmark(const char *label, size_t map_size,
			       int extra_flags, struct perf_counter *counters,
			       size_t counter_count)
{
	bool collect_perf = counters && counter_count > 0;

	flush_tlb();

	if (collect_perf) {
		enable_counters(counters, counter_count);
	}

	void *mapping = NULL;
	jit_kernel_fn fn = prepare_jit_kernel(map_size, extra_flags, &mapping);
	if (!fn) {
		if (collect_perf) {
			disable_counters(counters, counter_count);
		}
		printf("Benchmark: %s\n", label);
		printf("  status: skipped (unable to allocate executable mapping)\n");
		return;
	}

	uint64_t iterations = g_cfg.iterations;
	double elapsed = execute_code_kernel_mt(fn, iterations, g_cfg.threads);

	if (collect_perf) {
		disable_counters(counters, counter_count);
	}

	printf("Benchmark: %s\n", label);
	printf("  threads: %u\n", g_cfg.threads);
	printf("  iterations per-thread: %" PRIu64 "\n", iterations);
	printf("  duration: %.6f s\n", elapsed);

	struct tlb_stats itlb = { 0 };
	struct tlb_stats dtlb = { 0 };

	for (size_t i = 0; i < counter_count; ++i) {
		struct perf_counter *ctr = &counters[i];
		if (!ctr->available) {
			continue;
		}
		if (strcmp(ctr->name, "instructions") == 0) {
			double rate = elapsed > 0.0 ?
					      (double)ctr->value / elapsed :
					      0.0;
			printf("  instructions: %" PRIu64 " (%.3f G instr/s)\n",
			       ctr->value, rate / 1e9);
			continue;
		}

		if (is_itlb_label(ctr->name)) {
			record_tlb_counter(&itlb, ctr->name, ctr->value);
		} else if (is_dtlb_label(ctr->name)) {
			record_tlb_counter(&dtlb, ctr->name, ctr->value);
		}
	}

	print_tlb_stats("ITLB", itlb);
	print_tlb_stats("DTLB", dtlb);

	release_mapping(mapping, map_size);
}

/**
 * prime_buffer - Seed a buffer with deterministic pseudo-random values.
 * @buffer: Pointer to the buffer to initialise.
 * @elements: Number of 64-bit elements in the buffer.
 *
 * The initial contents drive both the read-side (source) and write-side
 * (destination) behaviours of the data benchmark.  A simple Weyl sequence keeps
 * the pattern dense, reproducible, and cheap to compute while ensuring that
 * consecutive elements are unlikely to collide.
 */
static void prime_buffer(uint64_t *buffer, size_t elements)
{
	for (size_t i = 0; i < elements; ++i) {
		buffer[i] = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
	}
}

/**
 * struct data_thread_args - Per-thread parameters for the memory benchmark.
 *
 * Encapsulates the per-worker view of the shared source/destination buffers,
 * loop bounds, and synchronisation primitives used by the read/modify/write
 * workload.
 */
struct data_thread_args {
	const uint64_t *src; /**< Shared source working-set pointer. */
	uint64_t *dst; /**< Shared destination working-set pointer. */
	size_t elements; /**< Total elements in each working-set. */
	uint64_t iterations; /**< Per-thread iteration budget. */
	unsigned thread_index; /**< Logical index of this worker. */
	unsigned thread_count; /**< Number of cooperating workers. */
	struct thread_start_sync *sync; /**< Synchronisation primitive. */
};

/**
 * data_thread_main - worker loop for the memory benchmark.
 * @arg: Pointer to struct data_thread_args describing the work.
 *
 * Return: NULL (the pthread exit value is unused).
 */
static void *data_thread_main(void *arg)
{
	struct data_thread_args *ctx = (struct data_thread_args *)arg;
	wait_for_start(ctx->sync);

	const unsigned stride = ctx->thread_count;
	uint64_t local = 0;
	for (uint64_t iter = 0; iter < ctx->iterations; ++iter) {
		for (size_t i = ctx->thread_index; i < ctx->elements;
		     i += stride) {
			uint64_t src_val = ctx->src[i];
			uint64_t dst_val = ctx->dst[i];
			/* Combine the source and destination cache lines to simulate
             * read/modify/write streams typical of memcpy-like workloads
             * with light computation layered on top. */
			uint64_t mixed = src_val ^ (dst_val + (iter + 1));
			mixed = (mixed * 2862933555777941757ULL) +
				3037000493ULL;
			ctx->dst[i] = mixed;
			local += mixed;
		}
	}

	__atomic_fetch_xor(&g_sink_u64, local, __ATOMIC_RELAXED);
	return NULL;
}

/**
 * execute_data_kernel_mt - launch the memory kernel across N threads.
 * @buffer: Shared working-set pointer.
 * @elements: Number of elements in @buffer.
 * @iterations: Per-thread iteration budget.
 * @thread_count: Number of worker threads to spawn.
 * @counters: Optional perf counters.
 * @counter_count: Number of counters provided.
 * @collect_perf: Whether to enable/sample the counters.
 *
 * Return: Elapsed time in seconds.
 */
static double execute_data_kernel_mt(const uint64_t *src, uint64_t *dst,
				     size_t elements, uint64_t iterations,
				     unsigned thread_count)
{
	struct thread_start_sync sync;
	thread_start_sync_init(&sync, thread_count);

	pthread_t *threads = calloc(thread_count, sizeof(*threads));
	struct data_thread_args *args = calloc(thread_count, sizeof(*args));
	if (!threads || !args) {
		die_message("Failed to allocate thread bookkeeping");
	}

	for (unsigned i = 0; i < thread_count; ++i) {
		/* Each worker receives its own starting offset into the shared buffers. */
		args[i].src = src;
		args[i].dst = dst;
		args[i].elements = elements;
		args[i].iterations = iterations;
		args[i].thread_index = i;
		args[i].thread_count = thread_count;
		args[i].sync = &sync;
		if (pthread_create(&threads[i], NULL, data_thread_main,
				   &args[i]) != 0) {
			die_errno("pthread_create (data)");
		}
	}

	pthread_barrier_wait(&sync.barrier);

	struct timespec start_ts = { 0 }, end_ts = { 0 };
	clock_gettime(CLOCK_MONOTONIC, &start_ts);
	atomic_store_explicit(&sync.go, true, memory_order_release);

	for (unsigned i = 0; i < thread_count; ++i) {
		pthread_join(threads[i], NULL);
	}
	clock_gettime(CLOCK_MONOTONIC, &end_ts);

	thread_start_sync_destroy(&sync);
	free(args);
	free(threads);

	return timespec_diff(&start_ts, &end_ts);
}

/**
 * run_data_benchmark - allocate and execute the memory benchmark.
 * @label: Benchmark name used in logs.
 * @map_size: Requested mapping size.
 * @extra_flags: Additional MAP_* flags for the allocation.
 * @counters: Perf counter array to use.
 * @counter_count: Number of entries in @counters.
 */
static void run_data_benchmark(const char *label, size_t map_size,
			       int extra_flags, struct perf_counter *counters,
			       size_t counter_count)
{
	bool collect_perf = counters && counter_count > 0;

	flush_tlb();

	uint64_t *src =
		map_buffer(map_size, PROT_READ | PROT_WRITE, extra_flags);
	if (!src) {
		printf("Benchmark: %s\n", label);
		printf("  status: skipped (unable to allocate buffer)\n");
		return;
	}

	size_t elements = map_size / sizeof(uint64_t);

	if (collect_perf) {
		enable_counters(counters, counter_count);
	}

	prime_buffer(src, elements);

	uint64_t *dst =
		map_buffer(map_size, PROT_READ | PROT_WRITE, extra_flags);
	if (!dst) {
		if (collect_perf) {
			disable_counters(counters, counter_count);
		}
		release_mapping(src, map_size);
		printf("Benchmark: %s\n", label);
		printf("  status: skipped (unable to allocate buffer)\n");
		return;
	}

	prime_buffer(dst, elements);

	/*
     * Each logical iteration reads one map-sized chunk from the source buffer
     * and writes the transformed results into the destination buffer.  The
     * two buffers ensure that every loop generates both load and store TLB
     * pressure, providing a closer approximation of real-world copy/update
     * kernels than a read-only stress test.
     */
	uint64_t iterations = select_data_iterations(map_size);
	double elapsed = execute_data_kernel_mt(src, dst, elements, iterations,
						g_cfg.threads);

	if (collect_perf) {
		disable_counters(counters, counter_count);
	}

	unsigned __int128 bytes_processed = (unsigned __int128)iterations *
					    (unsigned __int128)map_size * 2u;
	double bandwidth = elapsed > 0.0 ? (double)bytes_processed / elapsed :
					   0.0;

	printf("Benchmark: %s\n", label);
	printf("  threads: %u\n", g_cfg.threads);
	printf("  iterations per-thread: %" PRIu64 "\n", iterations);
	printf("  duration: %.6f s\n", elapsed);
	printf("  bytes processed: %.3Lf GiB\n",
	       (long double)bytes_processed / (1024.0L * 1024.0L * 1024.0L));
	printf("  bandwidth: %.3f GiB/s\n",
	       bandwidth / (1024.0 * 1024.0 * 1024.0));

	struct tlb_stats itlb = { 0 };
	struct tlb_stats dtlb = { 0 };
	for (size_t i = 0; i < counter_count; ++i) {
		struct perf_counter *ctr = &counters[i];
		if (!ctr->available) {
			continue;
		}
		if (is_itlb_label(ctr->name)) {
			record_tlb_counter(&itlb, ctr->name, ctr->value);
		} else if (is_dtlb_label(ctr->name)) {
			record_tlb_counter(&dtlb, ctr->name, ctr->value);
		}
	}

	print_tlb_stats("ITLB", itlb);
	print_tlb_stats("DTLB", dtlb);

	release_mapping(dst, map_size);
	release_mapping(src, map_size);
}

/**
 * init_code_counters - populate the perf counter array for instruction tests.
 * @counters: Array of perf counters to initialise.
 * @capacity: Maximum number of counters available in @counters.
 *
 * Return: Number of counters initialised.
 */
static size_t init_code_counters(struct perf_counter *counters, size_t capacity)
{
	size_t count = 0;

	if (capacity == 0) {
		return 0;
	}

	if (count < capacity) {
		init_counter(&counters[count++], "instructions",
			     PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
	}

	if (capacity - count < 3) {
		die_message("Insufficient counter slots for ITLB events");
	}

	init_named_cpu_counter(&counters[count++], "bp_l1_tlb_fetch_hit",
			       "bp_l1_tlb_fetch_hit");
	init_named_cpu_counter(&counters[count++], "bp_l1_tlb_miss_l2_hit",
			       "bp_l1_tlb_miss_l2_hit");
	init_named_cpu_counter(&counters[count++], "bp_l1_tlb_miss_l2_tlb_miss",
			       "bp_l1_tlb_miss_l2_tlb_miss");

	if (count < capacity) {
		init_cache_counter(&counters[count++], "dTLB-loads",
				   PERF_COUNT_HW_CACHE_DTLB,
				   PERF_COUNT_HW_CACHE_OP_READ,
				   PERF_COUNT_HW_CACHE_RESULT_ACCESS);
	}

	if (count < capacity) {
		init_cache_counter(&counters[count++], "dTLB-load-misses",
				   PERF_COUNT_HW_CACHE_DTLB,
				   PERF_COUNT_HW_CACHE_OP_READ,
				   PERF_COUNT_HW_CACHE_RESULT_MISS);
	}

	return count;
}

/**
 * init_data_counters - configure perf counters for the data benchmark.
 * @counters: Array of perf counters to initialise.
 * @capacity: Maximum number of counters available.
 *
 * Return: Number of counters initialised.
 */
static size_t init_data_counters(struct perf_counter *counters, size_t capacity)
{
	size_t count = 0;

	if (capacity == 0) {
		return 0;
	}

	if (capacity - count < 3) {
		die_message(
			"Insufficient counter slots for ITLB events (data phase)");
	}

	init_named_cpu_counter(&counters[count++], "bp_l1_tlb_fetch_hit",
			       "bp_l1_tlb_fetch_hit");
	init_named_cpu_counter(&counters[count++], "bp_l1_tlb_miss_l2_hit",
			       "bp_l1_tlb_miss_l2_hit");
	init_named_cpu_counter(&counters[count++], "bp_l1_tlb_miss_l2_tlb_miss",
			       "bp_l1_tlb_miss_l2_tlb_miss");

	if (count < capacity) {
		init_cache_counter(&counters[count++], "dTLB-loads",
				   PERF_COUNT_HW_CACHE_DTLB,
				   PERF_COUNT_HW_CACHE_OP_READ,
				   PERF_COUNT_HW_CACHE_RESULT_ACCESS);
	}

	if (count < capacity) {
		init_cache_counter(&counters[count++], "dTLB-load-misses",
				   PERF_COUNT_HW_CACHE_DTLB,
				   PERF_COUNT_HW_CACHE_OP_READ,
				   PERF_COUNT_HW_CACHE_RESULT_MISS);
	}

	return count;
}

/**
 * mapping_succeeds - probe helper to test whether a mapping size is available.
 * @size: Mapping size to probe.
 * @prot: Protection flags to use.
 * @extra_flags: Additional MAP_* flags.
 *
 * Return: true when the mapping succeeds.
 */
static bool mapping_succeeds(size_t size, int prot, int extra_flags)
{
	void *addr = mmap(NULL, size, prot,
			  MAP_PRIVATE | MAP_ANONYMOUS | extra_flags, -1, 0);
	if (addr == MAP_FAILED) {
		return false;
	}
	munmap(addr, size);
	return true;
}

/**
 * main - program entry point wiring the benchmarks together.
 * @argc: Argument count.
 * @argv: Argument vector.
 *
 * Return: 0 on success.
 */
int main(int argc, char **argv)
{
	config_init(&g_cfg);
	parse_args(argc, argv, &g_cfg);
	normalize_config(&g_cfg);

	bool run_code_regular = !g_cfg.skip_code && !g_cfg.skip_code_4k;
	bool run_code_huge = !g_cfg.skip_code && !g_cfg.skip_code_huge;
	bool run_any_code = run_code_regular || run_code_huge;

	struct perf_counter code_counters[7];
	size_t code_counter_count = 0;
	if (!g_cfg.disable_perf && run_any_code) {
		code_counter_count = init_code_counters(
			code_counters, ARRAY_SIZE(code_counters));
	}

	if (run_any_code) {
		size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
		size_t code_regular_size = 256UL * 1024 * 1024;
		if (code_regular_size % page_size) {
			code_regular_size =
				((code_regular_size / page_size) + 1) *
				page_size;
		}

		struct perf_counter *code_counter_ptr =
			code_counter_count ? code_counters : NULL;
		if (run_code_regular) {
			/*
             * The 4K code benchmark JITs a 256 MiB executable region filled
             * with pseudo-random basic blocks.  Using a large footprint keeps
             * the ITLB saturated even on systems with deep second-level TLBs
             * and exercises instruction cache replacement alongside the raw
             * translation machinery.
             */
			run_code_benchmark("code-4K", code_regular_size, 0,
					   code_counter_ptr,
					   code_counter_count);
		}

		if (run_code_huge) {
			bool huge_2mb_ok = mapping_succeeds(
				code_regular_size,
				PROT_READ | PROT_WRITE | PROT_EXEC,
				MAP_HUGETLB | MAP_HUGE_2MB);
			if (huge_2mb_ok) {
				run_code_benchmark("code-hugetlb-2MB",
						   code_regular_size,
						   MAP_HUGETLB | MAP_HUGE_2MB,
						   code_counter_ptr,
						   code_counter_count);
			} else {
				printf("Benchmark: code-hugetlb-2MB\n");
				printf("  status: skipped (2 MB huge pages unavailable)\n");
			}
		}
	}

	if (!g_cfg.disable_perf && run_any_code) {
		cleanup_counters(code_counters, code_counter_count);
	}

	bool run_data_regular = !g_cfg.skip_data && !g_cfg.skip_data_4k;
	bool run_data_huge = !g_cfg.skip_data && !g_cfg.skip_data_huge;
	bool run_any_data = run_data_regular || run_data_huge;

	struct perf_counter data_counters[5];
	size_t data_counter_count = 0;
	if (!g_cfg.disable_perf && run_any_data) {
		data_counter_count = init_data_counters(
			data_counters, ARRAY_SIZE(data_counters));
	}

	if (run_any_data) {
		size_t regular_size = g_cfg.regular_bytes;
		struct perf_counter *data_counter_ptr =
			data_counter_count ? data_counters : NULL;

		if (run_data_regular) {
			run_data_benchmark("data-4K", regular_size, 0,
					   data_counter_ptr,
					   data_counter_count);
		}

		if (run_data_huge) {
			bool huge_1gb_ok = mapping_succeeds(
				1UL << 30, PROT_READ | PROT_WRITE,
				MAP_HUGETLB | MAP_HUGE_1GB);
			if (huge_1gb_ok) {
				run_data_benchmark("data-hugetlb-1GB",
						   1UL << 30,
						   MAP_HUGETLB | MAP_HUGE_1GB,
						   data_counter_ptr,
						   data_counter_count);
			} else {
				printf("Benchmark: data-hugetlb-1GB\n");
				printf("  status: skipped (1 GB huge pages unavailable)\n");
			}
		}
	}

	if (!g_cfg.disable_perf && run_any_data) {
		cleanup_counters(data_counters, data_counter_count);
	}

	if (g_perf_permission_error) {
		fprintf(stderr,
			"Warning: perf_event_open denied (check /proc/sys/kernel/perf_event_paranoid).\n");
	}

	return 0;
}
