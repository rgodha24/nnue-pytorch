#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "lib/nnue_training_data_formats.h"
#include "training_data_loader_internal.h"

using namespace binpack;
using namespace chess;

// Test positions
struct TestCase {
    std::string name;
    std::string fen;
};

std::vector<TestCase> test_cases = {
  {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
  {"king_a1", "k7/8/8/8/8/8/8/7K w - - 0 1"},
  {"king_h1", "7k/8/8/8/8/8/8/K7 w - - 0 1"},
  {"king_a8", "K7/8/8/8/8/8/8/7k w - - 0 1"},
  {"king_h8", "7K/8/8/8/8/8/8/k7 w - - 0 1"},
  {"kings_opposite", "k7/8/8/8/8/8/8/K7 w - - 0 1"},
  {"complex", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"},
};

void print_features(const std::string& name, const std::string& fen) {
    // Create a TrainingDataEntry from FEN
    auto pos = Position::fromFen(fen);

    // Create a minimal entry (we only need the position)
    TrainingDataEntry entry;
    entry.pos    = pos;
    entry.score  = 0;
    entry.result = 0;
    entry.ply    = 0;

    // Get the feature extractor
    auto feature_set = get_feature("HalfKAv2_hm");
    if (!feature_set)
    {
        std::cerr << "Failed to get feature extractor" << std::endl;
        return;
    }

    int                max_features = feature_set->max_active_features();
    std::vector<int>   features(max_features);
    std::vector<float> values(max_features);

    std::cout << "\n===============================================" << std::endl;
    std::cout << "Test: " << name << std::endl;
    std::cout << "FEN: " << fen << std::endl;
    std::cout << "White King at: " << static_cast<int>(pos.kingSquare(Color::White)) << std::endl;
    std::cout << "Black King at: " << static_cast<int>(pos.kingSquare(Color::Black)) << std::endl;

    // White perspective
    std::cout << "\n--- White Perspective ---" << std::endl;
    auto [white_count, white_inputs] =
      feature_set->fill_features_sparse(entry, features.data(), values.data(), Color::White);
    std::cout << "Feature count: " << white_count << std::endl;
    std::cout << "Total inputs: " << white_inputs << std::endl;

    // Sort features for consistent output
    std::vector<int> white_features(features.begin(), features.begin() + white_count);
    std::sort(white_features.begin(), white_features.end());

    std::cout << "Features (sorted):" << std::endl;
    for (int i = 0; i < white_count; i++)
    {
        std::cout << "  [" << i << "]: " << white_features[i] << std::endl;
    }

    // Black perspective
    std::cout << "\n--- Black Perspective ---" << std::endl;
    auto [black_count, black_inputs] =
      feature_set->fill_features_sparse(entry, features.data(), values.data(), Color::Black);
    std::cout << "Feature count: " << black_count << std::endl;
    std::cout << "Total inputs: " << black_inputs << std::endl;

    // Sort features for consistent output
    std::vector<int> black_features(features.begin(), features.begin() + black_count);
    std::sort(black_features.begin(), black_features.end());

    std::cout << "Features (sorted):" << std::endl;
    for (int i = 0; i < black_count; i++)
    {
        std::cout << "  [" << i << "]: " << black_features[i] << std::endl;
    }

    // Check if indices are valid
    bool white_valid = true;
    for (int i = 0; i < white_count; i++)
    {
        if (white_features[i] < 0 || white_features[i] >= white_inputs)
        {
            std::cout << "  WARNING: White feature " << i << " out of range: " << white_features[i]
                      << std::endl;
            white_valid = false;
        }
    }
    bool black_valid = true;
    for (int i = 0; i < black_count; i++)
    {
        if (black_features[i] < 0 || black_features[i] >= black_inputs)
        {
            std::cout << "  WARNING: Black feature " << i << " out of range: " << black_features[i]
                      << std::endl;
            black_valid = false;
        }
    }

    if (white_valid && black_valid)
    {
        std::cout << "\n✓ All feature indices are valid" << std::endl;
    }
}

int main() {
    std::cout << "HalfKAv2_hm Feature Extractor Ground Truth Generator" << std::endl;
    std::cout << "====================================================" << std::endl;

    for (const auto& test : test_cases)
    {
        print_features(test.name, test.fen);
    }

    std::cout << "\n\nDone!" << std::endl;
    return 0;
}
