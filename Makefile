# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -std=c99 -O2 -Iinclude
LDFLAGS = 
LDLIBS = -lm  # Add other libraries here if needed (e.g., -lSDL2)

# Final executable name
TARGET = main

# Directories
SRCDIR = src
INCDIR = include
BUILDDIR = build
BINDIR = bin

# Sources and objects
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)

# Default target
all: directories $(BINDIR)/$(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILDDIR) $(BINDIR)

# Final executable linking
$(BINDIR)/$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) $(LDLIBS)

# Object file compilation
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILDDIR) $(BINDIR)

# Deep clean
distclean: clean
	rm -f *.o *~ core

# Debug build
debug: CFLAGS += -g -DDEBUG -O0
debug: all

# Release build (optimized)
release: CFLAGS += -O3 -DNDEBUG
release: all

# Install target
install: all
	@echo "Installing to /usr/local/bin..."
	@cp $(BINDIR)/$(TARGET) /usr/local/bin/ || echo "Installation failed - try with sudo"

# Uninstall target
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Project information
info:
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Target: $(TARGET)"

# Run the program
run: all
	./$(BINDIR)/$(TARGET)

# Valgrind memory check
memcheck: debug
	valgrind --leak-check=full --track-origins=yes ./$(BINDIR)/$(TARGET)

# Static analysis with cppcheck
analyze:
	cppcheck --enable=all --inconclusive -I$(INCDIR) $(SRCDIR)

# Code formatting (if you have clang-format)
format:
	find $(SRCDIR) $(INCDIR) -name "*.c" -o -name "*.h" | xargs clang-format -i

# Dependency generation (advanced feature)
depend:
	$(CC) -MM -I$(INCDIR) $(SOURCES) > $(BUILDDIR)/dependencies.mk

# Include dependencies if they exist
-include $(BUILDDIR)/dependencies.mk

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  debug     - Build with debug information"
	@echo "  release   - Build optimized for release"
	@echo "  run       - Build and run"
	@echo "  clean     - Remove build artifacts"
	@echo "  distclean - Complete cleanup"
	@echo "  memcheck  - Run with Valgrind memory checker"
	@echo "  analyze   - Static code analysis"
	@echo "  format    - Format source code"
	@echo "  depend    - Generate dependencies"
	@echo "  info      - Show project information"
	@echo "  install   - Install the executable"
	@echo "  uninstall - Uninstall the executable"

.PHONY: all clean distclean debug release run memcheck analyze format help info install uninstall directories depend
