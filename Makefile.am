AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS}
AM_CPPFLAGS = -Iinclude -fopenmp -DWITH_PNG -DWITH_JPEG
AM_CXXFLAGS = -fopenmp
bin_PROGRAMS = gdaisy
gdaisy_LDFLAGS = -lOpenCL -ljpeg -lpng
gdaisy_SOURCES = src/daisy/main.cpp src/daisy/oclDaisy.cpp \
                src/kutility/general.cpp src/kutility/corecv.cpp src/kutility/image_io_bmp.cpp \
                src/kutility/image_io_png.cpp src/kutility/image_io_jpeg.cpp \
                src/kutility/image_io_pnm.cpp src/kutility/image_manipulation.cpp \
                src/kutility/interaction.cpp \
                src/ocl/cachedConstructs.cpp src/ocl/cachedProgram.cpp
dist_noinst_SCRIPTS = autogen.sh
