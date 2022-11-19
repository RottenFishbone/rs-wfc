/// A simple 2D vector
#[derive(Copy, Clone, Debug)]
pub struct Vec2 {
    pub x: i32,
    pub y: i32,
}
impl Vec2 {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y } 
    }
}

/// The valid directions to compare cells in WFC 
pub enum Direction { 
    Up = 0,
    Down,
    Left,
    Right,
}
