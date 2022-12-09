#include <algorithm>
#include <iterator>

#include "cache.h"
#include "util.h"
#include "arch_predictor.h"


#define NUM_CORE 1
#define LLC_SETS 2048
#define LLC_WAYS 16

//3-bit RRIP counter
#define MAXDIST 32
uint64_t dist[LLC_SETS][LLC_WAYS];

#include "helper_function.h"


Dist_Predictor* dist_demand;    //2K entries, 5-bit counter per each entry

//Sampler components tracking cache history
#define SAMPLER_ENTRIES 2800
#define SAMPLER_HIST 8
#define SAMPLER_SETS SAMPLER_ENTRIES/SAMPLER_HIST

//History time
#define TIMER_SIZE 1024
uint64_t set_timer[LLC_SETS];   //64 sets, where 1 timer is used for every set

//Mathmatical functions needed for sampling set
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6) )  //Helper function to sample 64 sets for each core


// Initialize replacement state
void CACHE::initialize_replacement()
{
    cout << "Initialize arch replacement policy state" << endl;

    for (int i=0; i<LLC_SETS; i++) {
        for (int j=0; j<LLC_WAYS; j++) {
	    dist[i][j] = MAXDIST;
        }
        set_timer[i] = 0;
    }


    dist_demand = new Dist_Predictor();

    cout << "Finished initializing arch replacement policy state" << endl;
}

// Find replacement victim
// Return value should be 0 ~ 15 or 16 (bypass)
uint32_t CACHE::find_victim(uint32_t cpu, uint64_t instr_id, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type)
{


    //Find highest value
    uint64_t max_dist = 0;
    int32_t victim = -1;
    for(uint32_t i = 0; i < LLC_WAYS; i++){
        if(dist[set][i] >= max_dist){
            max_dist = dist[set][i];
            victim = i;
        }
    }

    return victim;
}







// called on every cache hit and cache fill
void CACHE::update_replacement_state(uint32_t cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t PC, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{

    dist_demand -> update_dist(PC,set_timer[set]);
    dist[set][way] = dist_demand -> get_dist(PC);
    set_timer[set] = (set_timer[set] + 1) % TIMER_SIZE;

}

void CACHE::replacement_final_stats() {}
