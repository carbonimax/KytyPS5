# Keep the existing Linux IWYU warnings non-fatal.
set(KYTY_IWYU_COMMON "-Xiwyu;--cxx17ns;-Qunused-arguments")
if (LINUX)
	set(KYTY_IWYU_STRICT "")
else()
	set(KYTY_IWYU_STRICT ";-Werror")
endif()

function(include_what_you_use target dirs)
  if (CLANG AND ("${target}" IN_LIST KYTY_IWYU))
    find_program (CLANG_IWYU_EXE NAMES "include-what-you-use")
    if (CLANG_IWYU_EXE)
		set_target_properties(${target} PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${CLANG_IWYU_EXE};${KYTY_IWYU_COMMON}${KYTY_IWYU_STRICT}")
    endif()
  endif()
endfunction()

function(include_what_you_use_with_mappings target dirs mappings)
  if (CLANG AND ("${target}" IN_LIST KYTY_IWYU))
    find_program (CLANG_IWYU_EXE NAMES "include-what-you-use")
    if (CLANG_IWYU_EXE)
		foreach(map ${mappings})
			list(APPEND mapdirs ";-Xiwyu;--mapping_file=${map}")
		endforeach()
		set_target_properties(${target} PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${CLANG_IWYU_EXE};${mapdirs};${KYTY_IWYU_COMMON}${KYTY_IWYU_STRICT}")
    endif()
  endif()
endfunction()

function(clang_tidy_check target config headers dirs)
  if (KYTY_ENABLE_CLANG_TIDY AND CLANG AND ("${target}" IN_LIST KYTY_CLANG_TIDY) AND NOT KYTY_CLANG_CL)
    find_program (CLANG_TIDY_EXE NAMES "clang-tidy")
    if (CLANG_TIDY_EXE)
		set(std_arg "-extra-arg=-std=c++${CMAKE_CXX_STANDARD}")
		foreach(dir ${dirs})
			list(APPEND incdirs "-extra-arg=-I${dir}")
		endforeach()
		foreach(header ${headers})
			list(APPEND filter "(${header}.*)")
		endforeach()
		string(REPLACE ";" "|" filter "${filter}")
		if ("${config}" STREQUAL "")
			set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-warnings-as-errors=*;-header-filter=${filter};${std_arg};${incdirs}")	
		else()		
			set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-config=${config};-warnings-as-errors=*;-header-filter=${filter};${std_arg};${incdirs}")	
		endif()
    endif()
  endif()
endfunction()

macro(config_compiler_and_linker)

set(KYTY_WARNINGS_ARE_ERRORS OFF)

set(KYTY_C_FLAGS "")
set(KYTY_CPP_FLAGS "")

# Apple's /usr/bin/ar does not understand @response-file syntax, and macOS has a
# large ARG_MAX so response files aren't needed. Note: the Ninja generator forces a
# response file whenever CMAKE_NINJA_FORCE_RESPONSE_FILE is *defined* (it tests
# definedness, not the value), so on Apple it must be fully unset, not set to 0.
if(APPLE)
	unset(CMAKE_NINJA_FORCE_RESPONSE_FILE CACHE)
else()
	SET(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")
endif()

# Host instruction set. Guest code runs natively on x86-64 hosts, so every such host has x86-64-v3
# (AVX2, BMI1, BMI2, FMA, F16C, LZCNT, MOVBE) and the emulator is built for it. Never on other hosts
# (the ARM64 port). KYTY_X86_64_V3=OFF keeps the baseline x86-64 target, e.g. for a binary translator
# that lacks one of these extensions.
set(KYTY_HOST_X86_64 FALSE)
if(APPLE)
	if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
	   (NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"))
		set(KYTY_HOST_X86_64 TRUE)
	endif()
elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
	if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(x64|x86_64|AMD64)$")
		set(KYTY_HOST_X86_64 TRUE)
	endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
	set(KYTY_HOST_X86_64 TRUE)
endif()
option(KYTY_X86_64_V3 "Build x86-64 hosts for x86-64-v3 (AVX2, BMI2, FMA)" ${KYTY_HOST_X86_64})
set(KYTY_TARGET_X86_64_V3 FALSE)
if(KYTY_X86_64_V3 AND KYTY_HOST_X86_64)
	set(KYTY_TARGET_X86_64_V3 TRUE)
elseif(KYTY_X86_64_V3)
	message(WARNING "KYTY_X86_64_V3 is ignored: the target is not x86-64")
endif()
message(STATUS "x86-64-v3 target: ${KYTY_TARGET_X86_64_V3}")

# No floating-point contraction on any target: a * b + c keeps both roundings.
if(KYTY_CLANG_CL)
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
	endif()
	
	string(REGEX REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
	string(REGEX REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
	string(REGEX REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	string(REGEX REPLACE "/MD" "/MT" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
	string(REGEX REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
	string(REGEX REPLACE "/MD" "/MT" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
			
	set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} /utf-8 /Oy- /wd4244 /wd4305 /wd4800 /wd4345")
	set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} /clang:-ffp-contract=off")
	if(KYTY_TARGET_X86_64_V3)
		set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} /clang:-march=x86-64-v3")
	endif()
  
	if(KYTY_WARNINGS_ARE_ERRORS)
		#set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} /WX")
	    add_compile_options(/WX)
	endif()

	add_compile_options("$<$<CONFIG:Release>:/O2>")
	add_compile_options("$<$<CONFIG:Release>:/DNDEBUG>")
	add_compile_options("$<$<CONFIG:RelWithDebInfo>:/O2>")
	add_compile_options("$<$<CONFIG:RelWithDebInfo>:/DNDEBUG>")

	set(KYTY_C_FLAGS "${KYTY_CPP_FLAGS}")
	
elseif(CLANG OR GCC)

	if (CLANG)
	    set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} -fno-rtti -fno-exceptions -fcolor-diagnostics -finput-charset=UTF-8 -fexec-charset=UTF-8 -g -fno-strict-aliasing -fno-omit-frame-pointer -Wall -fmessage-length=0")
	    if (WIN32)
	    	set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} -static")
	    endif()
	else()
		set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} -fno-exceptions -fdiagnostics-color=always -finput-charset=UTF-8 -fexec-charset=UTF-8 -static-libgcc -static-libstdc++ -g -fno-strict-aliasing -fno-omit-frame-pointer -Wall -Wno-unused-value -fmessage-length=0")
	endif()
	set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} -ffp-contract=off")
	if(KYTY_TARGET_X86_64_V3)
		set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} -march=x86-64-v3")
	endif()
	
    if(KYTY_WARNINGS_ARE_ERRORS)
        #set(KYTY_CPP_FLAGS "${KYTY_CPP_FLAGS} -Werror")
	add_compile_options(-Werror)
    endif()	

	add_link_options("-g")
	
	unset(CMAKE_CXX_STANDARD_LIBRARIES CACHE)
	unset(CMAKE_C_STANDARD_LIBRARIES CACHE)
	
	set(KYTY_C_FLAGS "${KYTY_CPP_FLAGS}")
	
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${KYTY_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KYTY_CPP_FLAGS}")

endmacro()
