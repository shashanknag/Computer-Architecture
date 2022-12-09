#include <iostream>

#include "cache.h"
#define HLEN 4096
#define IHLEN 8192
static uint64_t prev_addr = 0;
static int current_dist = 0;
static int prev_dist = 0;
static uint64_t prev_ip = 0;


static int hist_table[2*HLEN] = {0};
static int ip_hist_table[IHLEN] = {0};

void CACHE::prefetcher_initialize() { std::cout << "CPU " << cpu << " Mixed Stride LLC prefetcher" << endl; }

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in)
{

  // Compute the current distance
  int current_dist = (addr - prev_addr); 

  // Compute the hashes for the current distance and the previous distance
  int hash1 = current_dist%HLEN;
  int hash2 = prev_dist%HLEN;

  // Index into the history table based on the hash functions and prefetch the cache block
  if(current_dist >=0)  {
	  uint64_t pf_addr = addr + hist_table[hash1];  
	  prefetch_line(ip, addr, pf_addr, true, 0);}

  if(current_dist <0)  {
	  uint64_t pf_addr = addr + hist_table[2*(- hash1)];  
	  prefetch_line(ip, addr, pf_addr, true, 0);}

  // Update the distance history table based on the hash of the previous distance
  if(prev_dist >=0)  
	  hist_table[hash2] = current_dist;

  if(prev_dist <0)  
	  hist_table[2*(-hash2)] = current_dist;


  // Compute the hashes for the IP and the previous IP
  int hash3 = ip%IHLEN;
  int hash4 = prev_ip%IHLEN;

  // Compute the prefetch address by indexing into the IP-based history table using the computed hashes
  uint64_t pf_addr = addr + ip_hist_table[hash3];  
  prefetch_line(ip, addr, pf_addr, true, 0);

  // Update the IP-based history table with the current distance
  ip_hist_table[hash4] = current_dist;
  
  // Update the state variables
  prev_addr = addr;
  prev_dist = current_dist;
  prev_ip = ip;
  
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

uint32_t CACHE::prefetcher_cache_fill(uint64_t v_addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_v_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_final_stats() {}
