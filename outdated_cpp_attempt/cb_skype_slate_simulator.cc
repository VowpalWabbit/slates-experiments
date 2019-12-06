#include <iostream>

#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

#include "live_model.h"
#include "constants.h"
#include "parse_example_json.h"

#include "common.h"

namespace rl = reinforcement_learning;

std::string get_shared_features(context_t context)
{
  std::stringstream ss;
  ss << R"("User":{"platform":")" << to_string(std::get<0>(context)) << R"(","connection":")" << to_string(std::get<1>(context)) << R"(","region":")" << to_string(std::get<2>(context)) << R"("})";
  return ss.str();
}

std::string get_action_features(std::vector<actions_t> const& actions)
{
  std::stringstream ss;
  ss << R"("_multi":[)";
  std::string joiner = "";
  for (auto const& action : actions)
  {
    ss << joiner << R"({"A":{"x":"=)" << std::get<0>(action) << R"(","y":"=)" << std::get<1>(action) << R"(","z":"=)"
       << std::get<2>(action) << R"("}})";
    joiner = ",";
  }
  ss << "]";
  return ss.str();
}

int main()
{
  rl::api_status status;
  rl::utility::configuration config;
  config.set(rl::name::APP_ID, "cb_simulator_test");
  config.set(rl::name::OBSERVATION_SENDER_IMPLEMENTATION, rl::value::OBSERVATION_FILE_SENDER);
  config.set(rl::name::INTERACTION_SENDER_IMPLEMENTATION, rl::value::INTERACTION_FILE_SENDER);
  config.set(rl::name::DECISION_SENDER_IMPLEMENTATION, rl::value::INTERACTION_FILE_SENDER);
  config.set(rl::name::INITIAL_EPSILON, "1.0");
  config.set(rl::name::MODEL_SRC, rl::value::FILE_MODEL_DATA);
  config.set(rl::name::MODEL_FILE_NAME, "./input.model");

  std::map<std::pair<context_t, actions_t>, std::vector<float>> mapping;
  read_file("/mnt/c/w/repos/slate_sim/data/coarse/averaged.csv", mapping);

  auto const err_fn = [](const rl::api_status& status, void*) { std::cout << status.get_error_msg() << "\n"; };

  rl::live_model rl(config, err_fn);
  if (rl.init(&status) != rl::error_code::success)
  {
    std::cout << status.get_error_msg() << "\n";
    return -1;
  }

  float total_reward = 0.f;
  auto x_actions = get_unique_actions<0>(mapping);
  auto y_actions = get_unique_actions<1>(mapping);
  auto z_actions = get_unique_actions<2>(mapping);
  std::vector<actions_t> actions;
  actions.reserve(x_actions.size() * y_actions.size() * z_actions.size());
  for (auto& x : x_actions)
  {
    for (auto& y : y_actions)
    {
      for (auto& z : z_actions)
      {
        actions.push_back({x,y,z});
      }
    }
  }
  std::cout << "Total number of actions: " << actions.size() << std::endl;

  std::default_random_engine rd{0};
  std::mt19937 eng(rd());
  std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);

  auto const action_features = get_action_features(actions);
  std::array<std::string, 8> context_strings =
  {
    "{" + get_shared_features({platform::Mac, connection::wifi, region::CA}) + "," + action_features + "}",
    "{" + get_shared_features({platform::Mac, connection::wifi, region::US}) + "," + action_features + "}",
    "{" + get_shared_features({platform::Mac, connection::wired, region::CA}) + "," + action_features + "}",
    "{" + get_shared_features({platform::Mac, connection::wired, region::US}) + "," + action_features + "}",
    "{" + get_shared_features({platform::Windows, connection::wifi, region::CA}) + "," + action_features + "}",
    "{" + get_shared_features({platform::Windows, connection::wifi, region::US}) + "," + action_features + "}",
    "{" + get_shared_features({platform::Windows, connection::wired, region::CA}) + "," + action_features + "}",
    "{" + get_shared_features({platform::Windows, connection::wired, region::US}) + "," + action_features + "}"
  };

  std::array<context_t, 8> contexts =
  {
    context_t{platform::Mac, connection::wifi, region::CA},
    context_t{platform::Mac, connection::wifi, region::US},
    context_t{platform::Mac, connection::wired, region::CA},
    context_t{platform::Mac, connection::wired, region::US},
    context_t{platform::Windows, connection::wifi, region::CA},
    context_t{platform::Windows, connection::wifi, region::US},
    context_t{platform::Windows, connection::wired, region::CA},
    context_t{platform::Windows, connection::wired, region::US}
  };

  std::uniform_int_distribution<size_t> context_distribution(0, 7);

  constexpr size_t num_rounds = 1;
  for (size_t i = 1; i <= num_rounds; i++)
  {
    auto chosen_context = context_distribution(eng);
    auto const& context = context_strings[chosen_context];
    rl::ranking_response rr;

    if (rl.choose_rank(context.c_str(), rr, &status))
    {
      std::cerr << status.get_error_msg() << "\n";
      return -1;
    }

    size_t action_id;
    if (rr.get_chosen_action_id(action_id))
    {
      std::cout << status.get_error_msg() << "\n";
      return -1;
    }

    auto const& chosen_action = actions[action_id - 1];
    auto reward = -1 * get_reward(contexts[chosen_context], chosen_action, mapping);
    total_reward += reward;

    if (i % 10 == 0)
    {
      std::cout << "i: " << i << ", Avg reward: " << total_reward / i << ", this reward: " << reward << "\n";
    }

    if (rl.report_outcome(rr.get_event_id(), reward, &status))
    {
      std::cerr << status.get_error_msg() << "\n";
      return -1;
    }
  }

  std::cout << "Total iterations: " << num_rounds << ", Avg reward: " << total_reward / num_rounds << "\n";
}
