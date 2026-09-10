# No symbol of a dependency libddx swallowed leaves it, and nothing past the C
# runtime is needed to load it.
#   cmake -DNM=<nm> -DREADELF=<readelf> -DLIB=<libddx.so>
#         -DWHAT=<name> -DFORBID=<regex> -P check_exports.cmake
execute_process(COMMAND "${NM}" -D --defined-only "${LIB}"
        OUTPUT_VARIABLE symbols COMMAND_ERROR_IS_FATAL ANY)
string(REGEX MATCHALL "[^\n]*(${FORBID})[^\n]*" leaked "${symbols}")
if (leaked)
    list(LENGTH leaked count)
    list(SUBLIST leaked 0 5 sample)
    string(JOIN "\n  " sample ${sample})
    message(FATAL_ERROR "${count} ${WHAT} symbols exported from ${LIB}, such as:\n  ${sample}")
endif ()

# libdl: the OpenCL loader's dlopen lives there before glibc 2.34.
execute_process(COMMAND "${READELF}" -d "${LIB}"
        OUTPUT_VARIABLE dynamic COMMAND_ERROR_IS_FATAL ANY)
string(REGEX MATCHALL "\\(NEEDED\\)[^\n]*" needed "${dynamic}")
set(foreign "")
foreach (entry IN LISTS needed)
    string(REGEX REPLACE ".*\\[(.*)\\].*" "\\1" soname "${entry}")
    if (NOT soname MATCHES "^(libstdc\\+\\+|libc\\+\\+|libc\\+\\+abi|libunwind|libm|libgcc_s|libc|libdl|ld-linux)[.-]")
        list(APPEND foreign "${soname}")
    endif ()
endforeach ()
if (foreign)
    message(FATAL_ERROR "${LIB} needs libraries a client may not have: ${foreign}")
endif ()
