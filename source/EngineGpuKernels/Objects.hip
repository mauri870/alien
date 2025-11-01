#include "Objects.hip.h"

#include "Object.hip.h"

void Objects::init()
{
    cellPointers.init();
    cells.init();
    particles.init();
    particlePointers.init();
    auxiliaryData.init();
}

void Objects::free()
{
    cellPointers.free();
    cells.free();
    particles.free();
    particlePointers.free();
    auxiliaryData.free();
}

__device__ void Objects::saveNumEntries()
{
    cellPointers.saveNumEntries();
    particlePointers.saveNumEntries();
    cells.saveNumEntries();
    particles.saveNumEntries();
}
