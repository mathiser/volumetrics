use ndarray::Array3;
use crate::utils::generate_edge;

struct CoordComponent {
    idx: u16,
    spacing: f32
}
impl CoordComponent {
    fn physical_point(&self) -> f32 {
        f32::from(self.idx) * &self.spacing
    }
}
struct Coord {
    z: CoordComponent,
    y: CoordComponent,
    x: CoordComponent,
}
impl Coord{
    fn distance_to(&self, other_coord: &Coord) -> f32 {
        let z_delta: f32 = other_coord.z.physical_point() - &self.z.physical_point();
        let y_delta: f32 = other_coord.y.physical_point() - &self.y.physical_point();
        let x_delta: f32 = other_coord.x.physical_point() - &self.x.physical_point();
        let dist = z_delta.powf(2.0) + y_delta.powf(2.0) + x_delta.powf(2.0);
        let dist = dist.sqrt();
        dist
    }
}

pub struct HausdorffMap {
    ref_arr: Array3<bool>,
    other_arr: Array3<bool>,
    zyx_spacing: Vec<f32>,
    ref_coords: Vec<Coord>,
    other_coords: Vec<Coord>,
    distances: Vec<f32>
}

impl HausdorffMap {
    fn new(ref_arr: Array3<bool>,
           other_arr: Array3<bool>,
           zyx_spacing: Vec<f32>) -> HausdorffMap {
        HausdorffMap {
            ref_arr: generate_edge(ref_arr, false),
            other_arr: generate_edge(other_arr, false),
            zyx_spacing,
            ref_coords: Vec::new(),
            other_coords: Vec::new(),
            distances: Vec::new(),
        }
    }
    fn execute(&mut self) {
        for z in 0..self.ref_arr.shape()[0] {
            for y in 0..self.ref_arr.shape()[1] {
                for x in 0..self.ref_arr.shape()[2] {
                    if *self.ref_arr.get([z, y, x]).unwrap() {
                        let _ = &self.ref_coords.push(
                            Coord {
                                z: CoordComponent { idx: z as u16, spacing: *&self.zyx_spacing[0] },
                                y: CoordComponent { idx: y as u16, spacing: *&self.zyx_spacing[1] },
                                x: CoordComponent { idx: x as u16, spacing: *&self.zyx_spacing[2] },
                            });
                    }
                    if *self.other_arr.get([z, y, x]).unwrap() {
                        let _ = &self.other_coords.push(
                            Coord {
                                z: CoordComponent { idx: z as u16, spacing: *&self.zyx_spacing[0] },
                                y: CoordComponent { idx: y as u16, spacing: *&self.zyx_spacing[1] },
                                x: CoordComponent { idx: x as u16, spacing: *&self.zyx_spacing[2] },
                            });
                    }
                }
            }
        }
        for ref_coord in &self.ref_coords {
            let mut min_dist: f32 = 99999999.0;
            let mut dist: f32;
            for other_coord in &self.other_coords {
                dist = ref_coord.distance_to(&other_coord);
                if dist.lt(&min_dist) {
                    min_dist = dist;
                };
            }
            let _ = &self.distances.push(min_dist);
        }
    }
    fn hd(&self) -> f32{
        let mut max_dist: f32 = -1.0;
        for dist in &self.distances {
            if dist.gt(&max_dist) {
                max_dist = *dist;
            }
        };
        max_dist
    }

    fn hd_percentile(&mut self, percentile: f32) -> f32{
        assert!((percentile.le(&1.0) & percentile.ge(&0.0)));

        let mut sorted_distances = self.distances.clone();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let vec_len = sorted_distances.len() as f32;
        let perc_idx = percentile * vec_len;
        let perc_idx = perc_idx as usize;
        sorted_distances[perc_idx - 1]  // shift index to fit vector
    }

}
#[cfg(test)]
mod test_hausdorff_map {
    use ndarray::Array3;
    use crate::distance::HausdorffMap;

    #[test]
    fn test_hd() {
        let mut src = Array3::<u8>::zeros([2, 5, 5]);
        for z in 0..src.shape()[0] {
            for y in 0..src.shape()[1] {
                for x in 0..4 {
                    src[[z, y, x]] = 1;
                }
            }
        }
        let mut dst = Array3::<u8>::zeros([2, 5, 5]);
        for z in 0..src.shape()[0] {
            for y in 0..src.shape()[1] {
                for x in 1..5 {
                    dst[[z, y, x]] = 1;
                }
            }
        }

        let src = src.mapv(|x| (x != 0));
        let dst = dst.mapv(|x| (x != 0));
        let mut hd_map = HausdorffMap::new(src, dst, vec![2.0, 1.0, 1.0]);
        hd_map.execute();

        assert!(&hd_map.distances.len().eq(&hd_map.ref_coords.len()));
        assert!(&hd_map.hd().eq(&1.0));
    }

    #[test]
    fn test_hd_percentile() {
        let src = Array3::<u8>::from(vec![
            [[1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 0]],
            [[1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0]]]);

        let dst = Array3::<u8>::from(vec![
           [[0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0]],
           [[0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0]]]);

        let src = src.mapv(|x| (x != 0));
        let dst = dst.mapv(|x| (x != 0));

        let mut hd_map = HausdorffMap::new(src, dst, vec![2.0, 2.4, 1.1]);
        hd_map.execute();
        assert!(&hd_map.distances.len().eq(&hd_map.ref_coords.len()));
        assert!(&hd_map.hd_percentile(1.0).eq(&hd_map.hd()));
        assert!(&hd_map.hd_percentile(0.95).eq(&3.2557642));

    }
}