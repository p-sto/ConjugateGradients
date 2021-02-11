/* Simple logger code. Nothing fancy here.
 *
 * Author: Pawel Stoworowicz
 *
 * */

#ifndef LOGGER_H
#define LOGGER_H

#ifdef DEBUG
#define DDEBUG 1
#else
#define DDEBUG 0
#endif

#define LEVEL_DEBUG -1
#define LEVEL_INFO 0
#define LEVEL_WARNING 1
#define LEVEL_ERROR 2

#ifndef DEBUG_FLAG
#define DEBUG_FLAG 0
#endif

#include <time.h>
#include <stdlib.h>
#include "utils.h"
#include <string.h>

#define _LOG_FIL_NAME "execution.log"

#define logger_clean()\
    FILE *logger_fp;\
    logger_fp = fopen(_LOG_FIL_NAME, "w+");\
    fclose(logger_fp);\

#define log_to_file(buffer, args...){\
    FILE *logger_fp;\
    logger_fp = fopen(_LOG_FIL_NAME, "a");\
    fprintf(logger_fp, buffer, ##args);\
    fclose(logger_fp);\
}

/*GCC prints warning regarding args... notation - works fine for CLang!*/
#define GCC_COMPILER (defined(__GNUC__) && !defined(__clang__))
#if GCC_COMPILER
#pragma GCC diagnostic ignored "-Wformat-extra-args"
#endif

#define _LOG_DATA_STDERR(buffer, fmt, args...){\
    strcat(buffer, fmt); strcat(buffer, "\n"); fprintf(stderr, buffer, ##args); log_to_file(buffer, ##args);\
    }

#define _LOG_DATA_STDOUT(buffer, fmt, args...){\
    strcat(buffer, fmt); strcat(buffer, "\n"); fprintf(stdout, buffer, ##args); log_to_file(buffer, ##args);\
    }


#define logger(log_level, fmt, args...){\
    char buffer[1024*sizeof(char)];\
    struct tm* tm_info;\
    time_t timer;\
    time(&timer);\
    tm_info = localtime(&timer);\
    strftime(buffer, 1024*sizeof(char), "%T", tm_info);\
    switch(log_level){\
        case LEVEL_INFO: strcat(buffer, " [INFO] "); _LOG_DATA_STDOUT(buffer, fmt, ##args) break;\
        case LEVEL_WARNING: strcat(buffer, " [WARNING] "); _LOG_DATA_STDOUT(buffer, fmt, ##args) break;\
        case LEVEL_ERROR: strcat(buffer, " ***[ERROR]*** "); _LOG_DATA_STDERR(buffer, fmt, ##args) break;\
        case LEVEL_DEBUG: if(DDEBUG){strcat(buffer, " [DEBUG] "); _LOG_DATA_STDOUT(buffer, fmt, ##args);\
        break;}\
            else{break;}\
        default: strcat(buffer, " [ERROR] Unknown level: "); break;\
    }/*switch*/\
}/*logger*/
#if GCC_COMPILER
#pragma GCC diagnostic pop
#endif /*pragma*/
#endif /*LOGGER_H*/
