#include <iostream>
#include <memory>
#include <tuple>
#include <array>
#include <algorithm>

#include <live_model.h>
#include <configuration.h>
#include <event_builder.h>
#include <constants.h>
#include <random>

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

  feature_space get_features() const
  {
    feature_space shared_context;
    feat_namespace shared_ns("User");
    shared_ns.push_feature("id=" + _id);
    shared_ns.push_feature("major=" + _major);
    shared_ns.push_feature("hobby=" + _hobby);
    shared_ns.push_feature("favorite_character=" + _favorite_character);
    shared_context.push_namespace(shared_ns);
    return shared_context;
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

std::vector<feature_space> generate_actions(std::vector<std::string> const& action_names)
{
  std::vector<feature_space> actions;
  for(auto const& action : action_names)
  {
    actions.push_back({{{"Action", {{"topic=" + action}}}}});
  }
  return actions;
}

int main()
{
  rl::api_status status;
  rl::utility::configuration config;
  config.set(rl::name::APP_ID, "slate_simulator_test");
  config.set(rl::name::OBSERVATION_SENDER_IMPLEMENTATION, rl::value::OBSERVATION_FILE_SENDER);
  config.set(rl::name::INTERACTION_SENDER_IMPLEMENTATION, rl::value::INTERACTION_FILE_SENDER);
  config.set(rl::name::DECISION_SENDER_IMPLEMENTATION, rl::value::INTERACTION_FILE_SENDER);
  config.set(rl::name::INITIAL_EPSILON, "1.0");
  config.set(rl::name::MODEL_SRC, rl::value::FILE_MODEL_DATA);
  config.set(rl::name::MODEL_FILE_NAME, "./input.model");
  config.set(rl::name::CCB_SAMPLE_MODE, rl::value::SAMPLE_SINGLE);
  config.set(rl::name::MODEL_VW_INITIAL_COMMAND_LINE, "--ccb_explore_adf --slate --json --quiet --epsilon 1.0 --id N/A");

  auto const err_fn = [](const rl::api_status& status, void*){
    std::cout << status.get_error_msg() << "\n";
  };

  float total_reward = 0.f;
  std::unique_ptr<rl::live_model> rl = std::unique_ptr<rl::live_model>(new rl::live_model(config, err_fn));
  if (rl->init(&status) != rl::error_code::success)
  {
    std::cout << status.get_error_msg() << "\n";
    return -1;
  }

  std::vector<std::string> const action_names{"HerbGarden", "MachineLearning"};
  std::vector<float> const tp1{0.3f, 0.2f};
  std::vector<float> const tp2{0.1f, 0.4f};

  std::vector<person> const people{
      person("rnc", "engineering", "hiking", "spock", tp1), person("mk", "psychology", "kids", "7of9", tp2)};

  std::default_random_engine rd{0};
  std::mt19937 eng(rd());
  std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);
  std::uniform_int_distribution<size_t> context_distribution(0, people.size() - 1);

  constexpr size_t num_rounds = 10000;
  estimator est(rl.get());

  feature_space slate_context;
  feat_namespace shared_ns("Slate");
  shared_ns.push_feature("constant");
  slate_context.push_namespace(shared_ns);

  auto actions = generate_actions(action_names);

  problem_data<slate_problem_type> prob_data;
  prob_data.push_slot({slate_context, actions});

  constexpr auto num_iterations = 10000;
  for (auto i = 0; i < num_iterations; i++)
  {
    auto const& person = people[context_distribution(eng)];
    problem_event<slate_problem_type> prob;
    prob.shared_context(person.get_features());
    prob.event_problem(prob_data);

    auto ret = est.predict_and_log(prob);
    auto it = ret.begin();
    size_t chosen;
    it->get_chosen_action_id(chosen);

    auto const draw = click_distribution(eng);
    auto const outcome = person.get_outcome(chosen, draw);
    total_reward += outcome;
    if (rl->report_outcome(it->get_event_id(), outcome, &status))
    {
      std::cerr << status.get_error_msg() << "\n";
      return -1;
    }

    if(i%500 == 0)
    {
      std::cout << "i: " << i<< ", Avg reward: " << total_reward/i << ", this reward: " << outcome << "\n";
    }
  }
  std::cout << "Total iterations: " << num_iterations<< ", Avg reward: " << total_reward/num_iterations << "\n";
}
