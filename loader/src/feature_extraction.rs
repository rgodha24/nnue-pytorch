use sfbinpack::chess::{color::Color, coords::Square, piece::Piece, position::Position};

pub const NUM_SQ: usize = 64;
pub const NUM_PT: usize = 12;
pub const NUM_PLANES: usize = NUM_SQ * NUM_PT;
pub const INPUTS: usize = NUM_PLANES * NUM_SQ / 2;
pub const MAX_ACTIVE_FEATURES: usize = 32;

pub const KING_BUCKETS: [i32; 64] = [
    -1, -1, -1, -1, 31, 30, 29, 28, -1, -1, -1, -1, 27, 26, 25, 24, -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16, -1, -1, -1, -1, 15, 14, 13, 12, -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4, -1, -1, -1, -1, 3, 2, 1, 0,
];

#[derive(Clone, Debug, PartialEq)]
pub struct SparseRow {
    pub is_white: f32,
    pub outcome: f32,
    pub score: f32,
    pub white_count: usize,
    pub black_count: usize,
    pub white: [i32; MAX_ACTIVE_FEATURES],
    pub black: [i32; MAX_ACTIVE_FEATURES],
    pub white_values: [f32; MAX_ACTIVE_FEATURES],
    pub black_values: [f32; MAX_ACTIVE_FEATURES],
    pub psqt_indices: i32,
    pub layer_stack_indices: i32,
}

pub fn orient_flip(color: Color, sq: Square, ksq: Square) -> Square {
    let horizontal_flip = if (ksq.index() & 7) < 4 { 7 } else { 0 };
    let vertical_flip = match color {
        Color::White => 0,
        Color::Black => 56,
    };

    Square::new(sq.index() ^ horizontal_flip ^ vertical_flip)
}

pub fn feature_index(color: Color, ksq: Square, sq: Square, piece: Piece) -> i32 {
    let oriented_ksq = orient_flip(color, ksq, ksq);
    let piece_index = piece.piece_type().ordinal() as i32 * 2 + i32::from(piece.color() != color);
    let bucket = KING_BUCKETS[oriented_ksq.index() as usize];

    orient_flip(color, sq, ksq).index() as i32
        + piece_index * NUM_SQ as i32
        + bucket * NUM_PLANES as i32
}

pub fn extract_features_into(
    pos: &Position,
    color: Color,
    features: &mut [i32; MAX_ACTIVE_FEATURES],
    values: &mut [f32; MAX_ACTIVE_FEATURES],
) -> usize {
    let mut count = 0usize;
    let mut occupied = pos.occupied();
    let ksq = pos.king_sq(color);

    while occupied.bits() != 0 {
        let sq = occupied.pop();
        let piece = pos.piece_at(sq);

        values[count] = 1.0;
        features[count] = feature_index(color, ksq, sq, piece);
        count += 1;
    }

    count
}

pub fn build_sparse_row(pos: &Position, score: i32, result: i32) -> SparseRow {
    let mut row = SparseRow {
        is_white: if pos.side_to_move() == Color::White {
            1.0
        } else {
            0.0
        },
        outcome: (result as f32 + 1.0) / 2.0,
        score: score as f32,
        white_count: 0,
        black_count: 0,
        white: [-1; MAX_ACTIVE_FEATURES],
        black: [-1; MAX_ACTIVE_FEATURES],
        white_values: [0.0; MAX_ACTIVE_FEATURES],
        black_values: [0.0; MAX_ACTIVE_FEATURES],
        psqt_indices: ((pos.occupied().count() as i32) - 1) / 4,
        layer_stack_indices: ((pos.occupied().count() as i32) - 1) / 4,
    };

    row.white_count =
        extract_features_into(pos, Color::White, &mut row.white, &mut row.white_values);
    row.black_count =
        extract_features_into(pos, Color::Black, &mut row.black, &mut row.black_values);

    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn king_bucket_table_matches_reference_layout() {
        assert_eq!(KING_BUCKETS[4], 31);
        assert_eq!(KING_BUCKETS[7], 28);
        assert_eq!(KING_BUCKETS[60], 3);
        assert_eq!(KING_BUCKETS[63], 0);
    }

    #[test]
    fn black_vertical_flip_matches_cpp_layout() {
        assert_eq!(
            orient_flip(Color::Black, Square::E8, Square::E8),
            Square::E1
        );
        assert_eq!(
            orient_flip(Color::Black, Square::H8, Square::H8),
            Square::H1
        );
    }

    #[test]
    fn startpos_extracts_32_features_per_side() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("valid startpos FEN");
        let row = build_sparse_row(&pos, 0, 0);

        assert_eq!(row.white_count, 32);
        assert_eq!(row.black_count, 32);
        assert!(row.white.iter().all(|&feature| feature >= 0));
        assert!(row.black.iter().all(|&feature| feature >= 0));
    }

    #[test]
    fn sparse_row_sets_tail_sentinels_for_shorter_positions() {
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").expect("valid endgame FEN");
        let row = build_sparse_row(&pos, 12, 1);

        assert_eq!(row.white_count, 2);
        assert_eq!(row.black_count, 2);
        assert_eq!(row.white[2], -1);
        assert_eq!(row.black[2], -1);
        assert_eq!(row.white_values[2], 0.0);
        assert_eq!(row.black_values[2], 0.0);
    }
}
