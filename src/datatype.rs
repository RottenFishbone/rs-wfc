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
    Invalid,
}
impl From<u32> for Direction {
    fn from(i: u32) -> Self {
        match i {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => Direction::Invalid
        }
    }
}
impl From<Direction> for String {
    fn from(dir: Direction) -> Self {
        match dir {
            Direction::Up => "Up".into(),
            Direction::Down => "Down".into(),
            Direction::Left => "Left".into(),
            Direction::Right => "Right".into(),
            Direction::Invalid => "Invalid".into(),
        }
    }
}


#[derive(Debug)]
pub struct Map<T> {
    pub size: Vec2,
    pub data: Vec<T>,
    /// Neighbour list provides a lookup for neighbour indicies.
    // This could also be a Vec of tuples to avoid existence checks (not sure if faster)
    pub neighbour_list: Vec<[Option<usize>; 4]>,
}
impl<T> Map<T> {

    /// Allocates space for a new Map
    /// Neighbours are initialized on creation, 
    /// data can optionally be initialized through `init_val`
    pub fn new(size: Vec2, init_val: Option<T>) -> Self 
    where T: Clone {

        if size.x < 0 || size.y < 0 {
            panic!("Tried to create a negative sized map.");
        }

        let total_size: usize = (size.x * size.y) as usize;
        let mut data: Vec<T> = Vec::with_capacity(total_size); 
        let mut neighbour_list = Vec::with_capacity(total_size);
        
        let val = init_val.clone();
        // Populate the neighbour_list and potentially the data (if init_val is set)
        for i in 0..total_size {
            // Convert i into an i32 for Point calculations
            let i_int: i32 = i.try_into().expect("Map total size should not exceed max i32");
            
            if init_val.is_some() {
                data.push(val.clone().unwrap());
            }

            let position = Vec2::new( i_int % size.x, i_int / size.x );
            let mut neighbours = [None; 4];
            
            // Left
            if position.x > 0 {
                neighbours[Direction::Left as usize] = Some(i-1);    
            }
            // Right
            if position.x < size.x-1 {
                neighbours[Direction::Right as usize] = Some(i+1);
            }
            // Up
            if position.y > 0 {
                neighbours[Direction::Up as usize] = Some(i-size.x as usize);
            }
            // Down
            if position.y < size.y-1 {
                neighbours[Direction::Down as usize] = Some(i+size.x as usize);
            }

            neighbour_list.push(neighbours);
        }

        Self { size, data, neighbour_list } 
    }
}
// Implement Display for pretty printing of maps
impl<T> std::fmt::Display for Map<T> where T: std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        for (i, cell) in self.data.iter().enumerate() {
            if i % self.size.x as usize == 0 {
                output.push_str("\n");
            }
            output.push_str(&format!("{:?} ", *cell)[..]); 
        }
        write!(f, "{}", output)
    }
}

/// Tilemap is a map of indicies, used as input/output for WFC
pub type Tilemap = Map<i32>;
