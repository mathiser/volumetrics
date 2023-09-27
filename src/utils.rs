use ndarray::{Array3, s};
fn clip_value(val: &usize, min: &usize, max: &usize) -> usize {
    if val.lt(min) {
        return min.to_owned()
    } else if val.gt( max) {
        return max.to_owned()
    } else {
        return val.to_owned()
    }
}
pub fn generate_edge(arr: Array3<bool>, use_2d: bool) -> Array3<bool> {
    let threshold_2d: u8 = 9;
    let threshold_3d: u8 = 27;

    let arr = arr.mapv(|arr| u8::from(arr));
    let mut edge_arr = Array3::<u8>::zeros(arr.raw_dim());
    let zmin = 0;
    let zmax = arr.shape()[0];
    let ymin = 0;
    let ymax = arr.shape()[1];
    let xmin = 0;
    let xmax = arr.shape()[2];

    for z in zmin..zmax {
        for y in ymin..ymax {
            for x in xmin..xmax {
                let val = arr.get([z, y, x]).unwrap();
                if val.eq(&u8::from(1)) {
                    let slice_zmin = clip_value(&z.checked_sub(1).unwrap_or(0), &zmin, &zmax);
                    let slice_zmax = clip_value(&(&z + 2).to_owned(), &zmin, &zmax);
                    let slice_ymin = clip_value(&y.checked_sub(1).unwrap_or(0), &ymin, &ymax);
                    let slice_ymax = clip_value(&(&y + 2).to_owned(), &ymin, &ymax);
                    let slice_xmin = clip_value(&x.checked_sub(1).unwrap_or(0), &xmin, &xmax);
                    let slice_xmax = clip_value(&(&x + 2).to_owned(), &xmin, &xmax);

                    if use_2d {
                        let sum: u8 = arr.slice(s![
                            z,
                            slice_ymin..slice_ymax,
                            slice_xmin..slice_xmax])
                            .iter().sum();
                        if sum < threshold_2d {
                            edge_arr[[z, y, x]] = arr[[z, y, x]];
                        }
                    } else {
                        let sum: u8 = arr.slice(s![
                            slice_zmin..slice_zmax,
                            slice_ymin..slice_ymax,
                            slice_xmin..slice_xmax])
                            .iter().sum();
                        if sum < threshold_3d {
                            edge_arr[[z, y, x]] = arr[[z, y, x]];
                        }
                    }
                }
            }
        }
    }
    let edge_arr = edge_arr.mapv(|x| (x != 0));
    edge_arr
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_generate_edge_2d() {
        let mut src = Array3::zeros([3, 5, 5]);
        for z in 0..src.shape()[0]{
            for y in 1..4 {
                for x in 1..4 {
                    src[[z, y, x]] = 1;
                }
            }
        }
        let mut dst = src.clone();
        for z in 0..src.shape()[0]{
            for y in 2..3 {
                for x in 2..3 {
                    dst[[z, y, x]] = 0;
                }
            }
        }

        let src = src.mapv(|x| (x != 0));
        let dst = dst.mapv(|x| (x != 0));
        let edge = generate_edge(src.clone(), true);
        assert!(edge == dst)
    }
    #[test]
    fn test_generate_edge_3d() {
        let mut src = Array3::zeros([3, 5, 5]);
        for z in 0..src.shape()[0]{
            for y in 1..4 {
                for x in 1..4 {
                    src[[z, y, x]] = 1;
                }
            }
        }
        let mut dst = src.clone();
        for z in 1..(src.shape()[0]-1){
            for y in 2..3 {
                for x in 2..3 {
                    dst[[z, y, x]] = 0;
                }
            }
        }

        let src = src.mapv(|x| (x != 0));
        let dst = dst.mapv(|x| (x != 0));
        let edge = generate_edge(src.clone(), false);

        assert!(edge == dst)
    }
}

