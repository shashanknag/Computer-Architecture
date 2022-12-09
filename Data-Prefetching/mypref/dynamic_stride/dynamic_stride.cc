#include <iostream>

#include "cache.h"
#define HLEN 4096

static uint64_t prev_addr = 0;
static int current_dist = 0;
static int prev_dist = 0;


static int hist_table[2*HLEN] = {0};

void CACHE::prefetcher_initialize() { std::cout << "CPU " << cpu << " Dynamic Stride LLC prefetcher" << endl; }

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in)
{

  // Compute the current distance
  int current_dist = (addr - prev_addr); 

  // Index into the history table based on the current distance and prefetch the cache block
  if(current_dist >=0 && current_dist < HLEN)  {
	  uint64_t pf_addr = addr + hist_table[current_dist];  
	  prefetch_line(ip, addr, pf_addr, true, 0);}

  if(current_dist <0 && current_dist > -HLEN)  {
	  uint64_t pf_addr = addr + hist_table[2*(- current_dist)];  
	  prefetch_line(ip, addr, pf_addr, true, 0);}

  // Update the distance history table by indexing into it using the previous distance
  if(prev_dist >=0 && prev_dist < HLEN)  
	  hist_table[prev_dist] = current_dist;

  if(prev_dist <0 && prev_dist > -HLEN)  
	  hist_table[2*(-prev_dist)] = current_dist;

  // Update the state variables
  prev_addr = addr;
  prev_dist = current_dist;
  
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

uint32_t CACHE::prefetcher_cache_fill(uint64_t v_addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_v_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_final_stats() {}
