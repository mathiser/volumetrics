use ndarray::{Array, Ix3};
use crate::utils::generate_edge;
pub struct APL {
    ref_arr: Array<bool, Ix3>,
    other_arr: Array<bool, Ix3>,
    diff_count: u32,
    other_total_count: u32
}
impl APL {
    pub fn new(ref_arr: &Array<bool, Ix3>, other_arr: &Array<bool, Ix3>) -> APL {
        APL {
            ref_arr: generate_edge(ref_arr, true),
            other_arr: generate_edge(other_arr, true),
            diff_count: 0,
            other_total_count: 0,
        }
    }
    pub fn execute(&mut self) {
        let mut ref_val: &bool;
        let mut other_val: &bool;
        for z in 0..self.ref_arr.shape()[0] {
            for y in 0..self.ref_arr.shape()[1] {
                for x in 0..self.ref_arr.shape()[2] {
                    ref_val = self.ref_arr.get([z, y, x]).unwrap();
                    other_val = self.other_arr.get([z, y, x]).unwrap();
                    if *other_val {
                        self.other_total_count += 1;
                        if !*ref_val {
                            self.diff_count += 1;
                        }
                    }
                }
            }
        }
    }

    pub fn apl(&self) -> u32 {
        self.diff_count
    }
    fn apl_normalized(&self) -> f32 {
        (self.diff_count as f32) / (self.other_total_count as f32)
    }
}

#[cfg(test)]
mod test_apl {
    use ndarray::Array3;
    use crate::apl::APL;

    #[test]
    fn test_apl() {
        let src = Array3::<u8>::from(vec![
               [[1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0]],
               [[1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0]]]);

        let dst = Array3::<u8>::from(vec![
               [[1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0]],
               [[0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0]]]);

        let src = src.mapv(|x| (x != 0));
        let dst = dst.mapv(|x| (x != 0));

        let mut f = APL::new(&src, &dst);
        f.execute();
        println!("{}", &f.apl());

        assert!(&f.apl().eq(&7));
        assert!(&f.apl_normalized().eq(&0.3888889));
    }
}