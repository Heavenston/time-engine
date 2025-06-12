use std::ops::Deref;

/// Provides a ReadOnly wrapper around the given type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ro<T>(T);

impl<T> Ro<T> {
    pub fn new(val: T) -> Self {
        Self(val)
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> From<T> for Ro<T> {
    fn from(val: T) -> Self {
        Self::new(val)
    }
}

impl<T> Deref for Ro<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
