add_subdirectory(local)
if(TARGET umbp_common)
  add_subdirectory(distributed)
endif()
