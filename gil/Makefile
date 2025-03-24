# Makefile for compiling all .c files in the gil directory

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -g -lpthread

# Source files
SRCS = $(wildcard *.c)

# Object files
OBJS = $(SRCS:.c=.out)

# Default target
all: $(OBJS)

# Rule to compile .c files to .out files
%.out: %.c
	$(CC) $(CFLAGS) -o $@ $<

# Clean target to remove all .out files
clean:
	rm -f $(OBJS)