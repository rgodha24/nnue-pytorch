use nnue_loader::chess::position::Position;
use nnue_loader::feature_extraction::{build_sparse_row, INPUTS, MAX_ACTIVE_FEATURES};

#[path = "support/cpp_reference.rs"]
mod cpp_reference;

use cpp_reference::{default_binpack_path, CppReference, ReferenceBatch};

#[test]
fn curated_halfka_rows_match_cpp_reference() {
    let cpp = CppReference::load().expect("load C++ loader shared library for parity testing");
    let fens = curated_fens();
    let scores = make_scores(fens.len());
    let plies = make_plies(fens.len());
    let results = make_results(fens.len());

    let batch = cpp
        .halfka_batch_from_fens(&fens, &scores, &plies, &results)
        .expect("obtain C++ reference batch for curated FENs");

    assert_eq!(batch.num_inputs, INPUTS as i32);
    assert_eq!(batch.size, fens.len());
    assert_eq!(batch.max_active_features, MAX_ACTIVE_FEATURES);

    let mut white_total = 0i32;
    let mut black_total = 0i32;

    for row in 0..fens.len() {
        let pos = Position::from_fen(&fens[row]).expect("curated FEN should parse");
        let rust_row = build_sparse_row(&pos, scores[row], results[row]);
        assert_row_matches_cpp(&batch, row, &rust_row);
        white_total += rust_row.white_count as i32;
        black_total += rust_row.black_count as i32;
    }

    assert_eq!(batch.num_active_white_features, white_total);
    assert_eq!(batch.num_active_black_features, black_total);
}

#[test]
fn sampled_dataset_halfka_rows_match_cpp_reference() {
    let Some(binpack_path) = default_binpack_path() else {
        eprintln!("skipping dataset parity sample: no .binpack file found under nnue-data/");
        return;
    };

    let cpp = CppReference::load().expect("load C++ loader shared library for parity testing");
    let fens = cpp
        .sample_fens(&binpack_path, 64)
        .expect("sample FENs from the current C++ loader");
    assert!(
        !fens.is_empty(),
        "dataset sample should return at least one FEN"
    );

    let scores = make_scores(fens.len());
    let plies = make_plies(fens.len());
    let results = make_results(fens.len());

    let batch = cpp
        .halfka_batch_from_fens(&fens, &scores, &plies, &results)
        .expect("obtain C++ reference batch for sampled dataset FENs");

    let mut white_total = 0i32;
    let mut black_total = 0i32;

    for row in 0..fens.len() {
        let pos = Position::from_fen(&fens[row]).expect("sampled dataset FEN should parse");
        let rust_row = build_sparse_row(&pos, scores[row], results[row]);
        assert_row_matches_cpp(&batch, row, &rust_row);
        white_total += rust_row.white_count as i32;
        black_total += rust_row.black_count as i32;
    }

    assert_eq!(batch.num_active_white_features, white_total);
    assert_eq!(batch.num_active_black_features, black_total);
}

fn assert_row_matches_cpp(
    batch: &ReferenceBatch,
    row: usize,
    rust_row: &nnue_loader::feature_extraction::SparseRow,
) {
    let cpp_is_white = batch.is_white[row];
    let cpp_outcome = batch.outcome[row];
    let cpp_score = batch.score[row];
    let cpp_psqt = batch.psqt_indices[row];
    let cpp_layer_stack = batch.layer_stack_indices[row];

    assert_eq!(
        cpp_is_white, rust_row.is_white,
        "is_white mismatch at row {row}"
    );
    assert_eq!(
        cpp_outcome, rust_row.outcome,
        "outcome mismatch at row {row}"
    );
    assert_eq!(cpp_score, rust_row.score, "score mismatch at row {row}");
    assert_eq!(
        cpp_psqt, rust_row.psqt_indices,
        "psqt index mismatch at row {row}"
    );
    assert_eq!(
        cpp_layer_stack, rust_row.layer_stack_indices,
        "layer stack index mismatch at row {row}"
    );

    assert_eq!(
        batch.white_row(row),
        rust_row.white.as_slice(),
        "white feature row mismatch at row {row}"
    );
    assert_eq!(
        batch.black_row(row),
        rust_row.black.as_slice(),
        "black feature row mismatch at row {row}"
    );
    assert_eq!(
        batch.white_values_row(row),
        rust_row.white_values.as_slice(),
        "white value row mismatch at row {row}"
    );
    assert_eq!(
        batch.black_values_row(row),
        rust_row.black_values.as_slice(),
        "black value row mismatch at row {row}"
    );

    assert!(
        rust_row.white_count == count_active_features(batch.white_row(row)),
        "white active feature count mismatch at row {row}"
    );
    assert!(
        rust_row.black_count == count_active_features(batch.black_row(row)),
        "black active feature count mismatch at row {row}"
    );
}

fn count_active_features(features: &[i32]) -> usize {
    features
        .iter()
        .take_while(|&&feature| feature != -1)
        .count()
}

fn curated_fens() -> Vec<String> {
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "k7/8/8/8/8/8/8/7K w - - 0 1",
        "7k/8/8/8/8/8/8/K7 w - - 0 1",
        "K7/8/8/8/8/8/8/7k w - - 0 1",
        "7K/8/8/8/8/8/8/k7 w - - 0 1",
        "k7/8/8/8/8/8/8/K7 b - - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/K7 w - - 0 1",
        "4k3/8/8/8/8/8/8/7K w - - 0 1",
        "rnbq1bnr/ppppkppp/8/4p3/3P4/2N5/PPP1PPPP/R1BQKBNR b KQ - 3 4",
        "4k3/1q6/8/3P4/8/8/6Q1/4K3 w - - 0 1",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn make_scores(len: usize) -> Vec<i32> {
    (0..len)
        .map(|idx| match idx % 4 {
            0 => -320,
            1 => -17,
            2 => 0,
            _ => 245,
        })
        .collect()
}

fn make_plies(len: usize) -> Vec<i32> {
    (0..len).map(|idx| (idx as i32 * 7) + 1).collect()
}

fn make_results(len: usize) -> Vec<i32> {
    (0..len)
        .map(|idx| match idx % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect()
}
