
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive_where::derive_where(Default)]
pub struct RLEVec<T> {
    counts: Vec<usize>,
    data: Vec<T>,
    len: usize,
}

impl<T> RLEVec<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(&mut self, val: T)
        where T: PartialEq
    {
        self.len += 1;
        if self.data.last() == Some(&val) {
            let len = self.counts.len();
            self.counts[len - 1] += 1;
        }
        else {
            self.data.push(val);
            self.counts.push(1);
        }
    }

    pub fn iter(&self) -> RLEIterator<'_, T> {
        RLEIterator {
            vec: self,
            idx: 0,
            repeat: 0,
        }
    }
}

struct RLEIterator<'a, T> {
    vec: &'a RLEVec<T>,
    idx: usize,
    repeat: usize,
}

impl<'a, T> Iterator for RLEIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.vec.len() {
            return None;
        }

        let item = &self.vec.data[self.idx];

        self.repeat += 1;
        if self.repeat >= self.vec.counts[self.idx] {
            self.idx += 1;
            self.repeat = 0;
        }

        Some(item)
    }
}
