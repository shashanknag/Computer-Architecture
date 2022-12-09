/**
 * Title: CS6600, Computer Architecture, Assignment - 3
 * Author: Sai Gautham Ravipati (EE19B053), Shashank Nag (EE19B118) 
 * Description: Branch Prediction using SRNN of slice-size of 8
 **/

// Required for updating the value of theta
#define SPEED 18

#include <algorithm>
#include <array>
#include <bitset>
#include <deque>
#include <map>

#include "ooo_cpu.h"

template <typename T, std::size_t HISTLEN, std::size_t BITS>

class srnn
{
  T Y = 0;                            // Holds the intermediate variable to store the weights 
  T output = 0;                       // Holds the output computaion 
  
  std::array<T, HISTLEN> W = {};           // Stores the weights to compute input-activation
  std::array<T, 5> B = {};                 // Stores the bias, at the end of each sub-stage
  std::array<T, 4> I = {};                 // Stores the accumulated result after each sub-stage 
  std::array<T, HISTLEN - 1> U = {};       // Holds the weight of each input-activation 
  std::array<T, HISTLEN> A = {};           // Holds the weight of input-activations

public:
  // maximum and minimum weight values
  // Need to saturate weights according to the BIT_WIDTH
  constexpr static T max_weight = (1 << (BITS - 1)) - 1;
  constexpr static T min_weight = -(max_weight + 1);

  T predict(std::bitset<HISTLEN> history)
  {
    
    // Computes the input-activations based on the global history
    for (std::size_t i = 0; i < std::size(history); i++) {
      if (history[i])
        A[i] = W[i];
      else
        A[i] = -W[i];
    }

    // Computation of stage-1 operating on A[0] - A[7]
    I[0] = U[0] * A[0] + A[1];
    I[0] = U[1] * I[0] + A[2];
    I[0] = U[2] * I[0] + A[3];
    I[0] = U[3] * I[0] + A[4];
    I[0] = U[4] * I[0] + A[5];
    I[0] = U[5] * I[0] + A[6];
    I[0] = U[6] * I[0] + A[7];
    I[0] += B[0];  

    // Computation of stage-2 operating on A[8] - A[15]  
    I[1] = U[8] * A[8] + A[9];
    I[1] = U[9] * I[1] + A[10];
    I[1] = U[10] * I[1] + A[11];
    I[1] = U[11] * I[1] + A[12];
    I[1] = U[12] * I[1] + A[13];
    I[1] = U[13] * I[1] + A[14];
    I[1] = U[14] * I[1] + A[15];
    I[1] += B[1];

    // Computation of stage-3 operating on A[16] - A[23]
    I[2] = U[16] * A[16] + A[17];
    I[2] = U[17] * I[2] + A[18];
    I[2] = U[18] * I[2] + A[19];
    I[2] = U[19] * I[2] + A[20];
    I[2] = U[20] * I[2] + A[21];
    I[2] = U[21] * I[2] + A[22];
    I[2] = U[22] * I[2] + A[23];
    I[2] += B[2];

    // Computation of stage-4 operating on A[24] - A[31]
    I[3] = U[24] * A[24] + A[25];
    I[3] = U[25] * I[3] + A[26];
    I[3] = U[26] * I[3] + A[27];
    I[3] = U[27] * I[3] + A[28];
    I[3] = U[28] * I[3] + A[29];
    I[3] = U[29] * I[3] + A[30];
    I[3] = U[30] * I[3] + A[31];
    I[3] += B[3];

    // Using the results of intermediate stages to obtain the output 
    Y = U[7] * I[0] + I[1];
    Y = U[15] * Y + I[2];
    Y = U[23] * Y + I[3];
    Y += B[4];

    output =  (Y + I[0] + I[1] + I[2] + I[3]) / 5;           // Normalising the results using a skip-connection 
    
    return output;
  }

  // Updating the state for sub-sequent computations 
  void update(bool result, std::bitset<HISTLEN> history)
  {
    
    // Updating the bias-value based on the branch result
    if(result) 
      B[4] = std::min(B[4] + 1, max_weight);
    else
      B[4] = std::max(B[4] - 1, min_weight);

    /* For each weight and corresponding bit in the history register. 
     * Updates the i'th bit in the history positively if it correlates 
     * with this branch outcome.
     */
    auto upd_mask = result ? history : ~history; 

    // Updating the weights of first stage, using the global history mask                                                                                                                      
    for (std::size_t i = 0; i < std::size(upd_mask); i++) {
      if (upd_mask[i])
        W[i] = std::min(W[i] + 1, max_weight);
      else
        W[i] = std::max(W[i] - 1, min_weight);
    }

    // Updating the weigths of input-activation, by comparing input activation with result 
    for (std::size_t i = 0; i < std::size(upd_mask) - 1; i++) {
      if (upd_mask[i])
        U[i] = std::min(U[i] + 1, max_weight);
      else
        U[i] = std::max(1, min_weight);
    }

    /* Updating the bias at the end-of stage using appropriate result
     * Check if the result at the end-of respective stage is contributing 
     * positively and make an update decision accordingly.
     */
    if(result == (I[0] >= 0))
      B[0] = std::min(B[0] + 1, max_weight);
    else
      B[0] = std::max(B[0] - 1, min_weight);

    if(result == (I[1] >= 0))
      B[1] = std::min(B[1] + 1, max_weight);
    else
      B[1] = std::max(B[1] - 1, min_weight);

    if(result == (I[2] >= 0))
      B[2] = std::min(B[2] + 1, max_weight);
    else
      B[2] = std::max(B[2] - 1, min_weight);

    if(result == (I[3] >= 0))
      B[3] = std::min(B[3] + 1, max_weight);
    else
      B[3] = std::max(B[3] - 1, min_weight);

  }
};

constexpr std::size_t SRNN_HISTORY = 32; // History length for the global history shift register
constexpr std::size_t SRNN_BITS = 8;     // Number of bits per weight
constexpr std::size_t NUM_SRNNS = 4096;  // No. of entries in the history table 

constexpr int THETA = 1.73 * SRNN_HISTORY + 14; // Threshold for training

constexpr std::size_t NUM_UPDATE_ENTRIES = 100; // Size of buffer for keeping 'srnn_state' for update

/* 'srnn_state' - stores the branch prediction and keeps information
 * such as output and history needed for updating the srnn predictor
 */
struct srnn_state {
  uint64_t ip = 0;
  bool prediction = false;                     // prediction: 1 for taken, 0 for not taken
  int output = 0;                              // srnn output
  std::bitset<SRNN_HISTORY> history = 0; // value of the history register yielding this prediction
};

std::map<O3_CPU*, std::array<srnn<int, SRNN_HISTORY, SRNN_BITS>,
                             NUM_SRNNS>> srnns;             // table of srnns
std::map<O3_CPU*, std::deque<srnn_state>> srnn_state_buf;   // state for updating srnn predictor
std::map<O3_CPU*, std::bitset<SRNN_HISTORY>> spec_global_history; // speculative global history - updated by predictor
std::map<O3_CPU*, std::bitset<SRNN_HISTORY>> global_history;      // real global history - updated when the predictor is
                                                                        // updated

int direction[1];     // Used in hashing consider the direction of branch based on ip and target 
// Placeholder for dynamic threshold setting 
int theta[1];         
int tc[1]; 

void O3_CPU::initialize_branch_predictor() {
  tc[0] = 0;
  theta[0] = THETA; 
}

uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type)
{
  // Hash the address to get an index into the table of srnns
  direction[0] = predicted_target - ip;
  auto index = ip ^ (ip >> SRNN_HISTORY) ^ (int)(spec_global_history[this].to_ulong());
  index = index % NUM_SRNNS;
  index += direction[0] % NUM_SRNNS;
  auto output = srnns[this][index].predict(spec_global_history[this]);

  bool prediction = (output >= 0);

  // Record the various values needed to update the predictor
  srnn_state_buf[this].push_back({ip, prediction, output, spec_global_history[this]});
  if (std::size(srnn_state_buf[this]) > NUM_UPDATE_ENTRIES)
    srnn_state_buf[this].pop_front();

  // Update the speculative global history register
  spec_global_history[this] <<= 1;
  spec_global_history[this].set(0, prediction);
  return prediction;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
  auto state = std::find_if(std::begin(srnn_state_buf[this]), std::end(srnn_state_buf[this]), [ip](auto x) { return x.ip == ip; });
  if (state == std::end(srnn_state_buf[this]))
    return; // Skip update because state was lost

  auto [_ip, prediction, output, history] = *state;
  srnn_state_buf[this].erase(state);

  auto index = ip ^ (ip >> SRNN_HISTORY) ^ (int)(history.to_ulong());
  index = index % NUM_SRNNS;
  index += direction[0] % NUM_SRNNS; 

  // Update the real global history shift register
  global_history[this] <<= 1;
  global_history[this].set(0, taken);

  /* If this branch was mispredicted, restore the speculative history to the
   * last known real history.
   */
  if (prediction != taken)
    spec_global_history[this] = global_history[this];

  /* if the output of the srnn predictor is outside of the range
   * [-theta,theta] *and* the prediction was correct, then we don't need to
   * adjust the weights.
   */
  if ((output <= theta[0] && output >= -theta[0]) || (prediction != taken)){
    srnns[this][index].update(taken, history);
    int a = output <= 0 ? -output : output; 

    // Dynamic threshold setting from Seznec's O-GEHL paper
    if (prediction != taken) {
      // Increase theta after enough mispredictions
      tc[0]++;
      if (tc[0] >= SPEED) {
        theta[0]++;
        tc[0] = 0;
      }
    } else if (a < theta[cpu]) {
      // Decrease theta after enough weak but correct predictions
      tc[0]--;
      if (tc[0] <= -SPEED) {
        theta[0]--;
        tc[0] = 0;
      }
    }
  } 
}
