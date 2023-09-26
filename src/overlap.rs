use ndarray::prelude::*;

pub struct ConfusionMatrix {
    ref_arr: Array3<bool>, 
    other_arr: Array3<bool>,
    tpos: u16,
    tneg: u16,
    fpos: u16,
    fneg: u16,
}

impl ConfusionMatrix {
    fn new(ref_arr: Array3<bool>, other_arr: Array3<bool>) -> ConfusionMatrix{
        ConfusionMatrix{
            ref_arr: ref_arr,
            other_arr: other_arr,
            tpos: 0,
            tneg: 0,
            fpos: 0,
            fneg: 0,
        }
    }
    fn execute(&mut self){
        let mut ref_val: &bool = &false;
        let mut other_val: &bool = &false;

        for z in 0..self.ref_arr.shape()[0] {
            for y in 0..self.ref_arr.shape()[1] {
                for x in 0..self.ref_arr.shape()[2] {
                    ref_val = self.ref_arr.get([z, y, x]).unwrap();
                    other_val = self.other_arr.get([z, y, x]).unwrap();
                    println!("{} {}", ref_val, other_val);
                    if ref_val & ref_val.eq(other_val)  { self.tpos += 1; }  // True positive
                    else if !ref_val & ref_val.eq(other_val) { self.tneg += 1; } // true negative
                    else if !ref_val & ref_val.ne(other_val) { self.fpos += 1; } // false positive
                    else if ref_val & ref_val.ne(other_val) { self.fneg += 1; } // false negative
                    }
                }
            }
        println!("{} {} {} {}",&self.tpos, &self.tneg, &self.fpos, &self.fneg, )
        }
    fn dc(&self) -> f32 {
        2.0*f32::from(*&self.tpos)/f32::from(&self.fpos + &self.fneg + 2*&self.tpos)
        }
    fn jc(&self) -> f32 {
        f32::from(*&self.tpos)/f32::from(&self.tpos + &self.fneg + &self.fpos)
        }
}



#[cfg(test)]
mod test_overlap {
    use ndarray::Array3;
    use crate::overlap::ConfusionMatrix;

    #[test]
    fn test_dc() {
        let src = Array3::<u8>::ones([10, 10, 10]);
        let dst = Array3::<u8>::zeros([10, 10, 10]);

        let src = src.mapv(|x| (x != 0));
        let mut dst = dst.mapv(|x| (x != 0));

        for z in 0..5 {
            for y in 0..10 {
                for x in 0..10 {
                    dst[[z, y, x]] = true;
                }
            }
        }
        let mut cm = ConfusionMatrix::new(
            src.clone(),
            dst.clone()
        );
        println!("{}", dst);
        cm.execute();
        let dice = cm.dc();
        assert_eq!(dice, 0.6666667);

        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    dst[[z, y, x]] = true;
                }
            }
        }
        let mut cm = ConfusionMatrix::new(
            src.clone(),
            dst.clone()
        );
        cm.execute();
        let dice = cm.dc();
        assert_eq!(dice, 1.0);

        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    dst[[z, y, x]] = false;
                }
            }
        }
        let mut cm = ConfusionMatrix::new(
            src.clone(),
            dst.clone()
        );
        cm.execute();
        let dice = cm.dc();
        assert_eq!(dice, 0.0);
    }
    #[test]
    fn test_jc() {
        let src = Array3::<u8>::ones([10, 10, 10]);
        let dst = Array3::<u8>::zeros([10, 10, 10]);

        let src = src.mapv(|x| (x != 0));
        let mut dst = dst.mapv(|x| (x != 0));

        for z in 0..5 {
            for y in 0..10 {
                for x in 0..10 {
                    dst[[z, y, x]] = true;
                }
            }
        }
        let mut cm = ConfusionMatrix::new(
            src.clone(),
            dst.clone()
        );
        cm.execute();
        let jacc = cm.jc();
        assert_eq!(jacc, 0.5);

        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    dst[[z, y, x]] = true;
                }
            }
        }
        let mut cm = ConfusionMatrix::new(
            src.clone(),
            dst.clone()
        );
        cm.execute();
        let jacc = cm.jc();
        assert_eq!(jacc, 1.0);

        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    dst[[z, y, x]] = false;
                }
            }
        }
        let mut cm = ConfusionMatrix::new(
            src.clone(),
            dst.clone()
        );
        cm.execute();
        let jacc = cm.jc();
        assert_eq!(jacc, 0.0);
    }
}