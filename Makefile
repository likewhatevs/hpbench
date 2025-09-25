CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
CFLAGS += -pthread
LDFLAGS ?=

TARGET := hpbench

all: $(TARGET)

$(TARGET): hpbench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
