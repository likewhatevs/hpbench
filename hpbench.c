#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#if !defined(__x86_64__)
#error "hpbench currently requires x86_64 for the JIT code benchmark"
#endif

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
#ifndef MAP_HUGE_2GB
#define MAP_HUGE_2GB (31 << MAP_HUGE_SHIFT)
#endif

/**
 * ARRAY_SIZE - count the number of elements in a static array.
 */
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/** Hard cap on threads to prevent runaway configuration. */
static const unsigned MAX_THREADS = 128;

#define DEFAULT_TARGET_RUNTIME_SEC 0.25
#define DEFAULT_REGULAR_BYTES (64UL * 1024 * 1024) /* 64 MiB */

/**
 * struct bench_config - central configuration knob set via CLI/env defaults.
 * @threads: Number of worker threads to launch for each benchmark.
 * @target_runtime_sec: Desired per-benchmark runtime in seconds.
 * @regular_bytes: Working-set size for the regular-page data test.
 */
struct bench_config {
    unsigned threads;           /**< Worker thread count. */
    double target_runtime_sec;  /**< Target duration for calibrations and runs. */
    size_t regular_bytes;       /**< Regular memory benchmark working-set bytes. */
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
 * struct perf_counter - thin wrapper around a perf_event_open file descriptor.
 * @name: Human-readable name for logging.
 * @fd: perf_event file descriptor returned by the kernel.
 * @available: Flag indicating whether the counter initialised successfully.
 * @type: perf_event_attr type value (hardware, cache, etc.).
 * @config: perf_event_attr config value describing the event.
 * @value: Snapshot captured after disabling the counter.
 *
 * We keep the metadata close to the fd, mirroring how a kernel developer might
 * wrap hardware counters inside a small struct.
 */
struct perf_counter {
    const char *name; /**< Counter label for reports. */
    int fd;           /**< Open perf_event file descriptor or -1. */
    bool available;   /**< True when the counter is active/usable. */
    uint32_t type;    /**< perf_event_attr::type selector. */
    uint64_t config;  /**< perf_event_attr::config payload. */
    uint64_t value;   /**< Latched counter reading. */
};

/**
 * struct tlb_stats - helper struct for TLB counter availability/value pairs.
 * @has_access: True when an access counter was gathered.
 * @has_miss: True when a miss counter was gathered.
 * @accesses: Count of TLB accesses.
 * @misses: Count of TLB misses.
 */
struct tlb_stats {
    bool has_access;  /**< True when the access counter is available. */
    bool has_miss;    /**< True when the miss counter is available. */
    uint64_t accesses;/**< Total TLB accesses recorded. */
    uint64_t misses;  /**< Total TLB misses recorded. */
};

/**
 * struct thread_start_sync - synchronisation scaffold for worker start.
 * @barrier: Barrier used to arm every worker thread before launch.
 * @go: Atomic flag flipped by the main thread to release workers.
 *
 * This mirrors the "fire together" primitives you often see in kernel selftests.
 */
struct thread_start_sync {
    pthread_barrier_t barrier; /**< Rendezvous point for all threads. */
    atomic_bool go;            /**< Release flag toggled by the controller thread. */
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
            "  -s, --size <bytes>   Working-set size for regular data test\n"
            "  -h, --help           Show this help\n",
            prog, DEFAULT_TARGET_RUNTIME_SEC);
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
}

/**
 * parse_env_overrides - honour the legacy HPBENCH_THREADS knob.
 * @cfg: Configuration object to update.
 *
 * The CLI remains the primary source; the environment only fills in when a
 * thread count was not specified explicitly.
 */
static void parse_env_overrides(struct bench_config *cfg)
{
    const char *env = getenv("HPBENCH_THREADS");

    if (env && *env && cfg->threads == 0) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end && *end == '\0' && parsed > 0) {
            cfg->threads = (unsigned)parsed;
        }
    }
}

/**
 * parse_args - kernel-benchmark-style option parsing using getopt_long.
 * @argc: Argument count from main().
 * @argv: Argument vector from main().
 * @cfg: Configuration object to populate with CLI overrides.
 */
static void parse_args(int argc, char **argv, struct bench_config *cfg)
{
    static const struct option long_opts[] = {
        {"threads", required_argument, NULL, 't'},
        {"runtime", required_argument, NULL, 'r'},
        {"size",    required_argument, NULL, 's'},
        {"help",    no_argument,       NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    int opt;

    while ((opt = getopt_long(argc, argv, "t:r:s:h", long_opts, NULL)) != -1) {
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
static long sys_perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu,
                                int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/**
 * die_errno - bail out with errno preserved.
 * @msg: Context string to print alongside strerror.
 */
static void die_errno(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

/**
 * die_message - helper for hard logic failures.
 * @msg: Message to emit before exiting.
 */
static void die_message(const char *msg) {
    fputs(msg, stderr);
    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

/**
 * cpu_relax - architecture hint for busy-waits.
 */
static inline void cpu_relax(void) {
    __asm__ __volatile__("pause" ::: "memory");
}

/**
 * thread_start_sync_init - prepare the barrier + flag pair used to line up workers.
 * @sync: Synchronisation primitive to initialise.
 * @thread_count: Number of worker threads participating.
 */
static void thread_start_sync_init(struct thread_start_sync *sync, unsigned thread_count) {
    if (pthread_barrier_init(&sync->barrier, NULL, thread_count + 1) != 0) {
        die_errno("pthread_barrier_init");
    }
    atomic_init(&sync->go, false);
}

/**
 * thread_start_sync_destroy - release resources paired with init.
 * @sync: Synchronisation primitive to destroy.
 */
static void thread_start_sync_destroy(struct thread_start_sync *sync) {
    if (pthread_barrier_destroy(&sync->barrier) != 0) {
        die_errno("pthread_barrier_destroy");
    }
}

/**
 * wait_for_start - common entry point for worker threads.
 * @sync: Synchronisation primitive to wait on.
 */
static void wait_for_start(struct thread_start_sync *sync) {
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
                         uint32_t type, uint64_t config) {
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

static bool read_sysfs(const char *path, char *buf, size_t buf_len)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        return false;
    }
    size_t n = fread(buf, 1, buf_len - 1, f);
    fclose(f);
    if (n == 0) {
        buf[0] = '\0';
        return false;
    }
    buf[n] = '\0';
    return true;
}

static int read_sysfs_int(const char *path, int *value)
{
    char buf[64];
    if (!read_sysfs(path, buf, sizeof(buf))) {
        return -1;
    }
    char *end = NULL;
    long val = strtol(buf, &end, 0);
    if (end == buf) {
        return -1;
    }
    *value = (int)val;
    return 0;
}

static bool parse_hex_u64(const char *str, uint64_t *out)
{
    char *end = NULL;
    unsigned long long val = strtoull(str, &end, 0);
    if (end == str) {
        return false;
    }
    *out = (uint64_t)val;
    return true;
}

static uint64_t set_config_bits(uint64_t config, unsigned shift, unsigned width, uint64_t value)
{
    uint64_t mask = ((1ULL << width) - 1ULL) << shift;
    config &= ~mask;
    config |= (value << shift) & mask;
    return config;
}

static void lower_string(char *dst, size_t dst_len, const char *src)
{
    size_t i;
    for (i = 0; i + 1 < dst_len && src[i]; ++i) {
        dst[i] = (char)tolower((unsigned char)src[i]);
    }
    dst[i] = '\0';
}

static char *trim_ws(char *str)
{
    while (*str && isspace((unsigned char)*str)) {
        str++;
    }
    if (!*str) {
        return str;
    }
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        *end-- = '\0';
    }
    return str;
}

static bool apply_cpu_event_token(struct perf_event_attr *attr, const char *token)
{
    char work[128];
    strncpy(work, token, sizeof(work) - 1);
    work[sizeof(work) - 1] = '\0';
    char *eq = strchr(work, '=');
    char *key = work;
    char *value = NULL;
    if (eq) {
        *eq = '\0';
        value = eq + 1;
    }
    key = trim_ws(key);
    if (value) {
        value = trim_ws(value);
    }
    char lower_key[32];
    lower_string(lower_key, sizeof(lower_key), key);

    uint64_t parsed = 0;

    if (!strcmp(lower_key, "event")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        attr->config = set_config_bits(attr->config, 0, 8, parsed);
        return true;
    }
    if (!strcmp(lower_key, "umask")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        attr->config = set_config_bits(attr->config, 8, 8, parsed);
        return true;
    }
    if (!strcmp(lower_key, "cmask")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        attr->config = set_config_bits(attr->config, 24, 8, parsed);
        return true;
    }
    if (!strcmp(lower_key, "edge")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        if (parsed) {
            attr->config |= 1ULL << 18;
        }
        return true;
    }
    if (!strcmp(lower_key, "inv")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        if (parsed) {
            attr->config |= 1ULL << 23;
        }
        return true;
    }
    if (!strcmp(lower_key, "any")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        if (parsed) {
            attr->config |= 1ULL << 21;
        }
        return true;
    }
    if (!strcmp(lower_key, "config")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        attr->config = parsed;
        return true;
    }
    if (!strcmp(lower_key, "config1")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        attr->config1 = parsed;
        return true;
    }
    if (!strcmp(lower_key, "config2")) {
        if (!value || !parse_hex_u64(value, &parsed)) {
            return false;
        }
        attr->config2 = parsed;
        return true;
    }

    /* Ignore unrecognised tokens gracefully. */
    return true;
}

static bool init_counter_sysfs(struct perf_counter *ctr, const char *event_alias,
                               const char *source, const char *event_name)
{
    char path[PATH_MAX];
    char buf[512];

    memset(ctr, 0, sizeof(*ctr));
    ctr->name = event_alias;
    ctr->fd = -1;

    if (snprintf(path, sizeof(path), "/sys/bus/event_source/devices/%s/type", source) >= (int)sizeof(path)) {
        return false;
    }
    int source_type = 0;
    if (read_sysfs_int(path, &source_type) != 0) {
        return false;
    }

    if (snprintf(path, sizeof(path), "/sys/bus/event_source/devices/%s/events/%s", source, event_name) >= (int)sizeof(path)) {
        return false;
    }
    if (!read_sysfs(path, buf, sizeof(buf))) {
        return false;
    }

    char *spec = trim_ws(buf);
    if (*spec == '\0') {
        return false;
    }

    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = (uint32_t)source_type;
    attr.size = sizeof(attr);
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.inherit = 1;

    char *saveptr = NULL;
    for (char *token = strtok_r(spec, ",", &saveptr);
         token != NULL;
         token = strtok_r(NULL, ",", &saveptr)) {
        if (!apply_cpu_event_token(&attr, token)) {
            return false;
        }
    }

    int fd = (int)sys_perf_event_open(&attr, 0, -1, -1, 0);
    if (fd == -1) {
        return false;
    }

    ctr->type = attr.type;
    ctr->config = attr.config;
    ctr->fd = fd;
    ctr->available = true;

    return true;
}

/**
 * close_counter - helper to centralise fd teardown.
 * @ctr: Counter wrapper to close.
 */
static void close_counter(struct perf_counter *ctr) {
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
static void enable_counters(struct perf_counter *counters, size_t count) {
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
static void disable_counters(struct perf_counter *counters, size_t count) {
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
        uint64_t value = 0;
        ssize_t read_bytes = read(ctr->fd, &value, sizeof(value));
        if (read_bytes != (ssize_t)sizeof(value)) {
            ctr->available = false;
            close_counter(ctr);
            continue;
        }
        ctr->value = value;
    }
}

/**
 * cleanup_counters - blanket close helper used before program exit.
 * @counters: Array of counter wrappers.
 * @count: Number of entries in @counters.
 */
static void cleanup_counters(struct perf_counter *counters, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        close_counter(&counters[i]);
    }
}

/**
 * tlb_hits - defensive helper so we never show negative hit counts.
 * @accesses: Number of TLB accesses observed.
 * @misses: Number of TLB misses observed.
 *
 * Return: Derived hit count (accesses - misses) clamped at zero.
 */
static inline uint64_t tlb_hits(uint64_t accesses, uint64_t misses) {
    if (misses > accesses) {
        return 0;
    }
    return accesses - misses;
}

/**
 * timespec_diff - convert two struct timespec samples into seconds.
 * @start: Earlier timestamp.
 * @end: Later timestamp.
 *
 * Return: Difference in seconds as a double.
 */
static double timespec_diff(const struct timespec *start,
                            const struct timespec *end) {
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
        threads = (unsigned)onln;
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
 * map_buffer - mmap wrapper used for both executable and data buffers.
 * @length: Mapping length in bytes.
 * @prot: Protection flags (PROT_*).
 * @extra_flags: Additional MAP_* flags to OR in.
 *
 * Return: Pointer to the mapping or NULL on failure.
 */
static void *map_buffer(size_t length, int prot, int extra_flags) {
    void *addr = mmap(NULL, length, prot,
                      MAP_PRIVATE | MAP_ANONYMOUS | extra_flags, -1, 0);
    if (addr == MAP_FAILED) {
        return NULL;
    }
    return addr;
}

/**
 * release_mapping - munmap wrapper with NULL protection.
 * @addr: Mapping address (ignored if NULL).
 * @length: Mapping length.
 */
static void release_mapping(void *addr, size_t length) {
    if (addr) {
        munmap(addr, length);
    }
}

/**
 * jit_kernel_template - tiny hand-written x86_64 loop.
 *
 * Every thread jumps into this code so we can measure iTLB pressure of regular
 * vs huge pages.
 */
static const unsigned char jit_kernel_template[] = {
    /* mov rcx, rdi */
    0x48, 0x89, 0xf9,
    /* xor rax, rax */
    0x48, 0x31, 0xc0,
    /* test rcx, rcx */
    0x48, 0x85, 0xc9,
    /* je done */
    0x74, 0x15,
    /* loop: add rax, rcx */
    0x48, 0x01, 0xc8,
    /* add rax, rcx */
    0x48, 0x01, 0xc8,
    /* add rax, rcx */
    0x48, 0x01, 0xc8,
    /* dec rcx */
    0x48, 0xff, 0xc9,
    /* jne loop */
    0x75, 0xee,
    /* done: ret */
    0xc3
};

typedef uint64_t (*jit_kernel_fn)(uint64_t iterations);

/**
 * prepare_jit_kernel - allocate and populate an executable code buffer.
 * @map_size: Size of the executable mapping.
 * @extra_flags: Extra MAP_* flags to request (e.g. MAP_HUGETLB).
 * @mapping_out: Optional return of the raw mapping address.
 *
 * Return: Function pointer to the generated kernel or NULL on failure.
 */
static jit_kernel_fn prepare_jit_kernel(size_t map_size, int extra_flags, void **mapping_out) {
    void *mapping = map_buffer(map_size, PROT_READ | PROT_WRITE | PROT_EXEC, extra_flags);
    if (!mapping) {
        return NULL;
    }
    memcpy(mapping, jit_kernel_template, sizeof(jit_kernel_template));
    __builtin___clear_cache((char *)mapping,
                            (char *)mapping + sizeof(jit_kernel_template));
    if (mapping_out) {
        *mapping_out = mapping;
    }
    return (jit_kernel_fn)mapping;
}

/**
 * struct code_thread_args - per-thread bundle for instruction benchmark.
 * @fn: Function pointer to execute.
 * @iterations: Number of loop iterations to perform.
 * @sync: Shared synchronisation object.
 */
struct code_thread_args {
    jit_kernel_fn fn;                 /**< Executable kernel to run. */
    uint64_t iterations;              /**< Loop iterations assigned to the thread. */
    struct thread_start_sync *sync;   /**< Start-line synchronisation primitive. */
};

/**
 * code_thread_main - worker loop for instruction-path benchmarking.
 * @arg: Pointer to struct code_thread_args describing the work item.
 *
 * Return: NULL (the pthread exit value is unused).
 */
static void *code_thread_main(void *arg) {
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
                                     unsigned thread_count,
                                     struct perf_counter *counters, size_t counter_count,
                                     bool collect_perf) {
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
		if (pthread_create(&threads[i], NULL, code_thread_main, &args[i]) != 0) {
			die_errno("pthread_create (code)");
		}
	}

	if (collect_perf && counters) {
		/* Counters only become live once all threads are ready. */
		enable_counters(counters, counter_count);
	}

	pthread_barrier_wait(&sync.barrier);

	struct timespec start_ts = {0}, end_ts = {0};
	clock_gettime(CLOCK_MONOTONIC, &start_ts);
	/* Flipping the go-flag releases every worker more or less simultaneously. */
	atomic_store_explicit(&sync.go, true, memory_order_release);

	for (unsigned i = 0; i < thread_count; ++i) {
		pthread_join(threads[i], NULL);
	}
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

    if (collect_perf && counters) {
        disable_counters(counters, counter_count);
    }

    thread_start_sync_destroy(&sync);
    free(args);
    free(threads);

    return timespec_diff(&start_ts, &end_ts);
}

/**
 * calibrate_code_iterations - derive a per-thread iteration count.
 * @fn: Executable kernel to probe.
 *
 * Return: Iteration budget that roughly matches g_cfg.target_runtime_sec.
 */
static uint64_t calibrate_code_iterations(jit_kernel_fn fn) {
	uint64_t iterations = 1ULL << 18; /* start smaller with threading */
	for (int attempt = 0; attempt < 8; ++attempt) {
		/* Measure with the actual thread fan-out so we capture scheduling cost. */
		double elapsed = execute_code_kernel_mt(fn, iterations, g_cfg.threads,
									NULL, 0, false);
		if (elapsed >= g_cfg.target_runtime_sec * 0.6) {
			return iterations;
		}
		double scale = g_cfg.target_runtime_sec / (elapsed > 1e-9 ? elapsed : 1e-9);
        if (scale < 1.5) {
            scale = 1.5;
        }
        if (scale > 8.0) {
            scale = 8.0;
        }
        double next = (double)iterations * scale;
        if (next > (double)(UINT64_MAX / 2)) {
            break;
        }
		iterations = (uint64_t)next;
	}
	return iterations;
}

/**
 * run_code_benchmark - front-end for the instruction benchmark.
 * @label: Benchmark name used in logs.
 * @map_size: Size of the executable mapping to allocate.
 * @extra_flags: Additional MAP_* flags (e.g. MAP_HUGETLB).
 * @counters: Perf counter array to use for measurements.
 * @counter_count: Number of entries in @counters.
 */
static void run_code_benchmark(const char *label, size_t map_size, int extra_flags,
                               struct perf_counter *counters, size_t counter_count) {
    void *mapping = NULL;
	jit_kernel_fn fn = prepare_jit_kernel(map_size, extra_flags, &mapping);
	if (!fn) {
		printf("Benchmark: %s\n", label);
		printf("  status: skipped (unable to allocate executable mapping)\n");
		return;
	}

	/* Derive how much work keeps the CPUs busy for roughly g_cfg.target_runtime_sec. */
	uint64_t iterations = calibrate_code_iterations(fn);
		double elapsed = execute_code_kernel_mt(fn, iterations, g_cfg.threads,
									counters, counter_count, true);

	printf("Benchmark: %s\n", label);
    printf("  threads: %u\n", g_cfg.threads);
    printf("  iterations per-thread: %" PRIu64 "\n", iterations);
    printf("  duration: %.6f s\n", elapsed);

    for (size_t i = 0; i < counter_count; ++i) {
        struct perf_counter *ctr = &counters[i];
        if (!ctr->available) {
            continue;
        }
        if (strcmp(ctr->name, "instructions") == 0) {
            double rate = elapsed > 0.0 ? (double)ctr->value / elapsed : 0.0;
            printf("  instructions: %" PRIu64 " (%.3f G instr/s)\n",
                   ctr->value, rate / 1e9);
        }
    }

    struct tlb_stats itlb = {0};
    struct tlb_stats dtlb = {0};

    for (size_t i = 0; i < counter_count; ++i) {
        struct perf_counter *ctr = &counters[i];
        if (!ctr->available) {
            continue;
        }
        if (strcmp(ctr->name, "itlb_access") == 0) {
            itlb.has_access = true;
            itlb.accesses = ctr->value;
        } else if (strcmp(ctr->name, "itlb_miss") == 0) {
            itlb.has_miss = true;
            itlb.misses = ctr->value;
        } else if (strcmp(ctr->name, "dtlb_access") == 0) {
            dtlb.has_access = true;
            dtlb.accesses = ctr->value;
        } else if (strcmp(ctr->name, "dtlb_miss") == 0) {
            dtlb.has_miss = true;
            dtlb.misses = ctr->value;
        }
    }

    if (itlb.has_access && itlb.has_miss) {
        uint64_t hits = tlb_hits(itlb.accesses, itlb.misses);
        printf("  ITLB: %" PRIu64 " accesses, %" PRIu64 " hits, %" PRIu64 " misses\n",
               itlb.accesses,
               hits,
               itlb.misses);
    } else if (itlb.has_miss) {
        printf("  ITLB: misses only (%" PRIu64 " misses)\n",
               itlb.misses);
    } else if (itlb.has_access) {
        printf("  ITLB: accesses only (%" PRIu64 " accesses)\n",
               itlb.accesses);
    } else {
        printf("  ITLB: counters unavailable\n");
    }

    if (dtlb.has_access && dtlb.has_miss) {
        uint64_t hits = tlb_hits(dtlb.accesses, dtlb.misses);
        printf("  DTLB: %" PRIu64 " accesses, %" PRIu64 " hits, %" PRIu64 " misses\n",
               dtlb.accesses,
               hits,
               dtlb.misses);
    } else if (dtlb.has_miss) {
        printf("  DTLB: misses only (%" PRIu64 " misses)\n",
               dtlb.misses);
    } else if (dtlb.has_access) {
        printf("  DTLB: accesses only (%" PRIu64 " accesses)\n",
               dtlb.accesses);
    } else {
        printf("  DTLB: counters unavailable\n");
    }

    release_mapping(mapping, map_size);
}

/**
 * prime_buffer - seed the working-set with deterministic values.
 * @buffer: Pointer to the buffer to initialise.
 * @elements: Number of 64-bit elements in the buffer.
 */
static void prime_buffer(uint64_t *buffer, size_t elements) {
    for (size_t i = 0; i < elements; ++i) {
        buffer[i] = (uint64_t)i;
    }
}

/**
 * struct data_thread_args - per-thread parameters for memory benchmark.
 * @buffer: Pointer to the shared working-set.
 * @elements: Number of elements visible to the benchmark.
 * @iterations: Per-thread iteration count.
 * @thread_index: Thread's zero-based index.
 * @thread_count: Total number of cooperating threads.
 * @sync: Shared synchronisation primitive.
 */
struct data_thread_args {
    uint64_t *buffer;              /**< Shared working-set base pointer. */
    size_t elements;               /**< Total elements in the working-set. */
    uint64_t iterations;           /**< Per-thread iteration budget. */
    unsigned thread_index;         /**< Logical index of this worker. */
    unsigned thread_count;         /**< Number of cooperating workers. */
    struct thread_start_sync *sync;/**< Synchronisation primitive. */
};

/**
 * data_thread_main - worker loop for the memory benchmark.
 * @arg: Pointer to struct data_thread_args describing the work.
 *
 * Return: NULL (the pthread exit value is unused).
 */
static void *data_thread_main(void *arg) {
    struct data_thread_args *ctx = (struct data_thread_args *)arg;
    wait_for_start(ctx->sync);

    const unsigned stride = ctx->thread_count;
    uint64_t local = 0;
    for (uint64_t iter = 0; iter < ctx->iterations; ++iter) {
        for (size_t i = ctx->thread_index; i < ctx->elements; i += stride) {
            uint64_t value = ctx->buffer[i];
            value = (value * 2862933555777941757ULL) + 3037000493ULL;
            ctx->buffer[i] = value;
            local += value;
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
static double execute_data_kernel_mt(uint64_t *buffer, size_t elements, uint64_t iterations,
                                     unsigned thread_count,
                                     struct perf_counter *counters, size_t counter_count,
                                     bool collect_perf) {
    struct thread_start_sync sync;
    thread_start_sync_init(&sync, thread_count);

    pthread_t *threads = calloc(thread_count, sizeof(*threads));
    struct data_thread_args *args = calloc(thread_count, sizeof(*args));
    if (!threads || !args) {
        die_message("Failed to allocate thread bookkeeping");
    }

	for (unsigned i = 0; i < thread_count; ++i) {
		/* Each worker receives its own starting offset into the shared buffer. */
		args[i].buffer = buffer;
		args[i].elements = elements;
		args[i].iterations = iterations;
		args[i].thread_index = i;
		args[i].thread_count = thread_count;
        args[i].sync = &sync;
        if (pthread_create(&threads[i], NULL, data_thread_main, &args[i]) != 0) {
            die_errno("pthread_create (data)");
        }
    }

	if (collect_perf && counters) {
		enable_counters(counters, counter_count);
	}

	pthread_barrier_wait(&sync.barrier);

	struct timespec start_ts = {0}, end_ts = {0};
	clock_gettime(CLOCK_MONOTONIC, &start_ts);
	atomic_store_explicit(&sync.go, true, memory_order_release);

    for (unsigned i = 0; i < thread_count; ++i) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

    if (collect_perf && counters) {
        disable_counters(counters, counter_count);
    }

    thread_start_sync_destroy(&sync);
    free(args);
    free(threads);

    return timespec_diff(&start_ts, &end_ts);
}

/**
 * calibrate_data_iterations - derive iteration count for memory benchmark.
 * @buffer: Working-set pointer used for calibration.
 * @elements: Number of elements in the working-set.
 *
 * Return: Iteration budget targeting g_cfg.target_runtime_sec.
 */
static uint64_t calibrate_data_iterations(uint64_t *buffer, size_t elements) {
	uint64_t iterations = 1;
	for (int attempt = 0; attempt < 8; ++attempt) {
		/* Light-touch calibration so we land near the target runtime quickly. */
		double elapsed = execute_data_kernel_mt(buffer, elements, iterations,
										g_cfg.threads, NULL, 0, false);
		if (elapsed >= g_cfg.target_runtime_sec * 0.6) {
            return iterations;
        }
		double scale = g_cfg.target_runtime_sec / (elapsed > 1e-9 ? elapsed : 1e-9);
        if (scale < 1.5) {
            scale = 1.5;
        }
        if (scale > 4.0) {
            scale = 4.0;
        }
        double next = (double)iterations * scale;
        if (next > (double)(UINT64_MAX / 2)) {
            break;
        }
        iterations = (uint64_t)next;
        if (iterations == 0) {
            iterations = UINT64_MAX;
            break;
        }
    }
    return iterations;
}

/**
 * run_data_benchmark - allocate, calibrate and execute the memory benchmark.
 * @label: Benchmark name used in logs.
 * @map_size: Requested mapping size.
 * @extra_flags: Additional MAP_* flags for the allocation.
 * @counters: Perf counter array to use.
 * @counter_count: Number of entries in @counters.
 */
static void run_data_benchmark(const char *label, size_t map_size, int extra_flags,
                               struct perf_counter *counters, size_t counter_count) {
    void *mapping = map_buffer(map_size, PROT_READ | PROT_WRITE, extra_flags);
    if (!mapping) {
        printf("Benchmark: %s\n", label);
        printf("  status: skipped (unable to allocate buffer)\n");
        return;
    }

	uint64_t *buffer = (uint64_t *)mapping;
	size_t elements = map_size / sizeof(uint64_t);
	prime_buffer(buffer, elements);

	uint64_t iterations = calibrate_data_iterations(buffer, elements);
	/* Re-prime after calibration to provide identical starting state. */
	prime_buffer(buffer, elements);

	/* Now perform the real measured run. */
	double elapsed = execute_data_kernel_mt(buffer, elements, iterations,
									g_cfg.threads, counters, counter_count, true);

    unsigned __int128 bytes_processed =
        (unsigned __int128)iterations * (unsigned __int128)map_size * 2u;
    double bandwidth = elapsed > 0.0 ? (double)bytes_processed / elapsed : 0.0;

    printf("Benchmark: %s\n", label);
    printf("  threads: %u\n", g_cfg.threads);
    printf("  iterations per-thread: %" PRIu64 "\n", iterations);
    printf("  duration: %.6f s\n", elapsed);
    printf("  bytes processed: %.3Lf GiB\n",
           (long double)bytes_processed / (1024.0L * 1024.0L * 1024.0L));
    printf("  bandwidth: %.3f GiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));

    struct tlb_stats dtlb = {0};
    for (size_t i = 0; i < counter_count; ++i) {
        struct perf_counter *ctr = &counters[i];
        if (!ctr->available) {
            continue;
        }
        if (strcmp(ctr->name, "dtlb_access") == 0) {
            dtlb.has_access = true;
            dtlb.accesses = ctr->value;
        } else if (strcmp(ctr->name, "dtlb_miss") == 0) {
            dtlb.has_miss = true;
            dtlb.misses = ctr->value;
        }
    }

    if (dtlb.has_access && dtlb.has_miss) {
        uint64_t hits = tlb_hits(dtlb.accesses, dtlb.misses);
        printf("  DTLB: %" PRIu64 " accesses, %" PRIu64 " hits, %" PRIu64 " misses\n",
               dtlb.accesses,
               hits,
               dtlb.misses);
    } else if (dtlb.has_miss) {
        printf("  DTLB: misses only (%" PRIu64 " misses)\n",
               dtlb.misses);
    } else if (dtlb.has_access) {
        printf("  DTLB: accesses only (%" PRIu64 " accesses)\n",
               dtlb.accesses);
    } else {
        printf("  DTLB: counters unavailable\n");
    }

    release_mapping(mapping, map_size);
}

/**
 * init_code_counters - populate the perf counter array for instruction tests.
 * @counters: Array of perf counters to initialise.
 */
static void init_code_counters(struct perf_counter *counters) {
    init_counter(&counters[0], "instructions", PERF_TYPE_HARDWARE,
                 PERF_COUNT_HW_INSTRUCTIONS);
    if (!init_counter_sysfs(&counters[1], "itlb_access", "cpu", "iTLB-loads")) {
        counters[1].name = "itlb_access";
    }
    if (!init_counter_sysfs(&counters[2], "itlb_miss", "cpu", "iTLB-load-misses")) {
        counters[2].name = "itlb_miss";
    }
    if (!init_counter_sysfs(&counters[3], "dtlb_access", "cpu", "dTLB-loads")) {
        counters[3].name = "dtlb_access";
    }
    if (!init_counter_sysfs(&counters[4], "dtlb_miss", "cpu", "dTLB-load-misses")) {
        counters[4].name = "dtlb_miss";
    }
}

/**
 * init_data_counters - configure perf counters for the data benchmark.
 * @counters: Array of perf counters to initialise.
 */
static void init_data_counters(struct perf_counter *counters) {
    if (!init_counter_sysfs(&counters[0], "dtlb_access", "cpu", "dTLB-loads")) {
        counters[0].name = "dtlb_access";
    }
    if (!init_counter_sysfs(&counters[1], "dtlb_miss", "cpu", "dTLB-load-misses")) {
        counters[1].name = "dtlb_miss";
    }
}

/**
 * mapping_succeeds - probe helper to test whether a mapping size is available.
 * @size: Mapping size to probe.
 * @prot: Protection flags to use.
 * @extra_flags: Additional MAP_* flags.
 *
 * Return: true when the mapping succeeds.
 */
static bool mapping_succeeds(size_t size, int prot, int extra_flags) {
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
    parse_env_overrides(&g_cfg);
    normalize_config(&g_cfg);

	struct perf_counter code_counters[5];
	init_code_counters(code_counters);

	size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
	run_code_benchmark("code-regular", page_size, 0,
			       code_counters, ARRAY_SIZE(code_counters));

	bool huge_2mb_ok = mapping_succeeds(2UL * 1024 * 1024,
					 PROT_READ | PROT_WRITE | PROT_EXEC,
					 MAP_HUGETLB | MAP_HUGE_2MB);
    if (huge_2mb_ok) {
        run_code_benchmark("code-hugetlb-2MB", 2UL * 1024 * 1024,
                           MAP_HUGETLB | MAP_HUGE_2MB,
                           code_counters, ARRAY_SIZE(code_counters));
    } else {
        printf("Benchmark: code-hugetlb-2MB\n");
        printf("  status: skipped (2 MB huge pages unavailable)\n");
    }

    cleanup_counters(code_counters, ARRAY_SIZE(code_counters));

	struct perf_counter data_counters[2];
	init_data_counters(data_counters);

	size_t regular_size = g_cfg.regular_bytes;
	run_data_benchmark("data-regular", regular_size, 0,
                       data_counters, ARRAY_SIZE(data_counters));

    bool huge_1gb_ok = mapping_succeeds(1UL << 30, PROT_READ | PROT_WRITE,
                                        MAP_HUGETLB | MAP_HUGE_1GB);
    if (huge_1gb_ok) {
        run_data_benchmark("data-hugetlb-1GB", 1UL << 30,
                           MAP_HUGETLB | MAP_HUGE_1GB,
                           data_counters, ARRAY_SIZE(data_counters));
    } else {
        printf("Benchmark: data-hugetlb-1GB\n");
        printf("  status: skipped (1 GB huge pages unavailable)\n");
    }

    bool huge_2gb_ok = mapping_succeeds(2UL << 30, PROT_READ | PROT_WRITE,
                                        MAP_HUGETLB | MAP_HUGE_2GB);
    if (huge_2gb_ok) {
        run_data_benchmark("data-hugetlb-2GB", 2UL << 30,
                           MAP_HUGETLB | MAP_HUGE_2GB,
                           data_counters, ARRAY_SIZE(data_counters));
    } else {
        printf("Benchmark: data-hugetlb-2GB\n");
        printf("  status: skipped (2 GB huge pages unavailable)\n");
    }

    cleanup_counters(data_counters, ARRAY_SIZE(data_counters));

    if (g_perf_permission_error) {
        fprintf(stderr,
                "Warning: perf_event_open denied (check /proc/sys/kernel/perf_event_paranoid).\n");
    }

    return 0;
}
