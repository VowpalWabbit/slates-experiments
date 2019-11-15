#include <iostream>

#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

#include "live_model.h"
#include "constants.h"
#include "parse_example_json.h"

namespace rl = reinforcement_learning;

struct person
{
  person(std::string id, std::string major, std::string hobby, std::string favorite_character, std::vector<float> dist)
      : _id(id), _major(major), _hobby(hobby), _favorite_character(favorite_character), _topic_click_probability(dist)
  {
  }

  std::string _id;
  std::string _major;
  std::string _hobby;
  std::string _favorite_character;
  std::vector<float> _topic_click_probability;

  std::string get_features() const
  {
    std::stringstream ss;
    ss << R"("User":{"id":")" << _id << R"(","major":")" << _major << R"(","hobby":")" << _hobby
       << R"(","favorite_character":")" << _favorite_character << R"("})";
    return ss.str();
  }

  float get_outcome(size_t chosen_action, float draw) const
  {
    auto click_prob = _topic_click_probability[chosen_action];
    if (draw <= click_prob)
    {
      return 1.0;
    }
    else
    {
      return 0.0;
    }
  }
};

std::string get_action_features(std::vector<std::string> const& actions)
{
  std::stringstream ss;
  ss << R"("_multi":[)";
  std::string joiner = "";
  for (auto const& action : actions)
  {
    ss << joiner << R"({"A":{"topic":")" << action << R"("}})";
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

  auto const err_fn = [](const rl::api_status& status, void*) { std::cout << status.get_error_msg() << "\n"; };

  rl::live_model rl(config, err_fn);
  if (rl.init(&status) != rl::error_code::success)
  {
    std::cout << status.get_error_msg() << "\n";
    return -1;
  }

  std::vector<std::string> const actions{"HerbGarden", "MachineLearning"};
  std::vector<float> const tp1{0.3f, 0.2f};
  std::vector<float> const tp2{0.1f, 0.4f};

  std::vector<person> const people{
      person("rnc", "engineering", "hiking", "spock", tp1), person("mk", "psychology", "kids", "7of9", tp2)};

  std::default_random_engine rd{0};
  std::mt19937 eng(rd());
  std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> context_distribution(0, people.size() - 1);

  constexpr size_t num_rounds = 10000;
  auto const action_features = get_action_features(actions);
  float reward = 0.f;
  for (size_t i = 1; i <= num_rounds; i++)
  {
    auto const& person = people[context_distribution(eng)];
    std::string const context = "{" +  person.get_features() +","+ action_features +"}";
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

    auto const draw = click_distribution(eng);
    auto const outcome = person.get_outcome(action_id, draw);
    reward += outcome;
    if (rl.report_outcome(rr.get_event_id(), outcome, &status))
    {
      std::cerr << status.get_error_msg() << "\n";
      return -1;
    }
  }
  std::cout << "Total iterations: " << num_rounds<< ", Avg reward: " << (reward / (float)num_rounds) << "\n";
}
