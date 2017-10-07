TARGET = prog
LIBS = -lm
CC = gcc
CFLAGS = -g -Wall -std=c11 -O2

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst src/%c.o, $(wildcard src/*.c))
HEADERS = $(wildcard src/*.h)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)

