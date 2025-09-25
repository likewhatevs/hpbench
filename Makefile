CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
CFLAGS += -pthread
LDFLAGS ?=

TARGET := hpbench
TEST_SCRIPT := validate_perf.py

all: $(TARGET)

$(TARGET): hpbench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: $(TARGET) $(TEST_SCRIPT)
	./$(TEST_SCRIPT) --binary ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean test
