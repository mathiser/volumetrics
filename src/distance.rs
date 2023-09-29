use ndarray::{Array, Ix3};
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
        let dist = &z_delta.powf(2.0) + &y_delta.powf(2.0) + &x_delta.powf(2.0);
        let dist = dist.sqrt();
        dist
    }
}

pub struct HausdorffMapDirected {
    ref_arr: Array<bool, Ix3>,
    other_arr: Array<bool, Ix3>,
    zyx_spacing: Vec<f32>,
    ref_coords: Vec<Coord>,
    other_coords: Vec<Coord>,
    distances: Vec<f32>
}

impl HausdorffMapDirected {
    pub fn new(ref_arr: &Array<bool, Ix3>,
           other_arr: &Array<bool, Ix3>,
           zyx_spacing: Vec<f32>) -> HausdorffMapDirected {
        HausdorffMapDirected {
            ref_arr: generate_edge(ref_arr, false),
            other_arr: generate_edge(other_arr, false),
            zyx_spacing,
            ref_coords: Vec::new(),
            other_coords: Vec::new(),
            distances: Vec::new(),
        }
    }
    pub fn execute(&mut self) {
        for z in 0..self.ref_arr.shape()[0] {
            for y in 0..self.ref_arr.shape()[1] {
                for x in 0..self.ref_arr.shape()[2] {
                    if *self.ref_arr.get([z, y, x]).unwrap() {
                        let _ = &self.ref_coords.push(
                            Coord {
                                z: CoordComponent { idx: z as u16, spacing: self.zyx_spacing[0] },
                                y: CoordComponent { idx: y as u16, spacing: self.zyx_spacing[1] },
                                x: CoordComponent { idx: x as u16, spacing: self.zyx_spacing[2] },
                            });
                    }
                    if *self.other_arr.get([z, y, x]).unwrap() {
                        let _ = &self.other_coords.push(
                            Coord {
                                z: CoordComponent { idx: z as u16, spacing: self.zyx_spacing[0] },
                                y: CoordComponent { idx: y as u16, spacing: self.zyx_spacing[1] },
                                x: CoordComponent { idx: x as u16, spacing: self.zyx_spacing[2] },
                            });
                    }
                }
            }
        }
        for ref_coord in &self.ref_coords {
            let mut min_dist: f32 = self.ref_coords[0].clone();
            let mut dist: f32;
            for other_coord in &self.other_coords {
                dist = ref_coord.distance_to(&other_coord);
                //dist = other_coord.distance_to(&ref_coord);

                if dist.lt(&min_dist) {
                    min_dist = dist;
                };
            }
            let _ = &self.distances.push(min_dist);
        }
    }
    pub fn hd(&self) -> f32{
        let mut max_dist: f32 = -1.0;
        for dist in &self.distances {
            if dist.gt(&max_dist) {
                max_dist = *dist;
            }
        };
        max_dist
    }
    pub fn asd(&self) -> f32{
        let mut sum: f32 = 0.0;
        for dist in &self.distances {
            sum += *dist;
            }
        &sum / self.distances.len() as f32
    }

    pub fn hd_percentile(&mut self, percentile: &f32) -> f32{
        assert!((percentile.le(&1.0) & percentile.ge(&0.0)));

        let mut sorted_distances = self.distances.clone();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let vec_len = (sorted_distances.len() - 1) as f32;  // shift index to fit vector
        let perc_idx = percentile * vec_len;
        let perc_idx = perc_idx as usize;
        sorted_distances[perc_idx]
    }

    pub fn surface_dc(&self, tolerance: &f32) -> f32 {
        let mut within: u32 = 0;
        for d in &self.distances{
            if d.le(&tolerance){
                within += 1;
            }
        }
        (within as f64 / self.distances.len() as f64) as f32
    }
}

pub struct HausdorffMapUndirected {
    ref_to_other_map: HausdorffMapDirected,
    other_to_ref_map: HausdorffMapDirected
}

impl HausdorffMapUndirected {
    pub fn new(ref_arr: &Array<bool, Ix3>,
               other_arr: &Array<bool, Ix3>,
               zyx_spacing: Vec<f32>) -> HausdorffMapUndirected {
        HausdorffMapUndirected {
            ref_to_other_map: HausdorffMapDirected::new(
                &ref_arr,
                &other_arr,
                zyx_spacing.clone()
            ),
            other_to_ref_map: HausdorffMapDirected::new(
                &other_arr,
                &ref_arr,
                zyx_spacing.clone()
            ),
        }
    }
    pub fn execute(&mut self) {
        let _ = &self.ref_to_other_map.execute();
        let _ = &self.other_to_ref_map.execute();
    }
    pub fn avg_hd(&self) -> f32 {
        (self.ref_to_other_map.hd() + self.other_to_ref_map.hd()) / 2 as f32
    }
    pub fn hd(&self) -> f32 {
        let ref_hd = self.ref_to_other_map.hd();
        let other_hd = self.other_to_ref_map.hd();
        if ref_hd.gt(&other_hd){
            ref_hd
        } else {
            other_hd
        }
    }
    pub fn assd(&self) -> f32 {
        (self.ref_to_other_map.asd() + self.other_to_ref_map.asd()) / 2 as f32
    }
    pub fn avg_hd_percentile(&mut self, percentile: &f32) -> f32{
        assert!((percentile.le(&1.0) & percentile.ge(&0.0)));
        (self.ref_to_other_map.hd_percentile(percentile) + self.other_to_ref_map.hd_percentile(percentile)) / 2 as f32
    }
    pub fn hd_percentile(&mut self, percentile: &f32) -> f32{
        assert!((percentile.le(&1.0) & percentile.ge(&0.0)));
        let ref_hd = self.ref_to_other_map.hd_percentile(percentile);
        let other_hd = self.other_to_ref_map.hd_percentile(percentile);
        if ref_hd.gt(&other_hd){
            ref_hd
        } else {
            other_hd
        }
    }
}
#[cfg(test)]
mod test_hausdorff_map {
    use ndarray::{Array, Array3, Ix3};
    use crate::distance::{HausdorffMapDirected, HausdorffMapUndirected};

    fn generate_src_dst_arrays() -> (Array<bool, Ix3>, Array<bool, Ix3>){
        let src = Array3::<u8>::from(vec![
               [[1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0]],
            [[1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
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
                [0, 1, 1, 1, 1]]]);

        let src = src.mapv(|x| (x != 0)) as Array<bool, Ix3>;
        let dst = dst.mapv(|x| (x != 0)) as Array<bool, Ix3>;
        (src, dst)
    }

    #[test]
    fn test_directed_hd() {
        let (src, dst) = generate_src_dst_arrays();
        let mut hd_map = HausdorffMapDirected::new(&src, &dst, vec![1.0, 2.0, 3.0]);
        hd_map.execute();

        assert!(&hd_map.distances.len().eq(&hd_map.ref_coords.len()));

        assert!(&hd_map.hd().eq(&1.0));
    }

    #[test]
    fn test_directed_hd_percentile() {
        let (src, dst) = generate_src_dst_arrays();

        let mut hd_map = HausdorffMapDirected::new(&src, &dst, vec![1.0, 2.0, 3.0]);
        hd_map.execute();
        assert!(&hd_map.distances.len().eq(&hd_map.ref_coords.len()));

        assert!(&hd_map.hd_percentile(&1.0).eq(&hd_map.hd()));
        assert!(&hd_map.hd_percentile(&0.95).eq(&1.0));
        assert!(&hd_map.hd_percentile(&0.0).eq(&0.0));
    }
    #[test]
    fn test_surface_dc() {
        let (src, dst) = generate_src_dst_arrays();
        let mut hd_map = HausdorffMapDirected::new(&src, &dst, vec![3.09, 4.1231, 2.343]);
        hd_map.execute();

        assert!(&hd_map.distances.len().eq(&hd_map.ref_coords.len()));
        assert!(&hd_map.surface_dc(&1.0).eq(&0.7894737));
        assert!(&hd_map.surface_dc(&2.0).eq(&0.7894737));
        assert!(&hd_map.surface_dc(&3.0).eq(&1.0));
    }
    #[test]
    fn test_undirected_hd_percentile() {
        let (src, dst) = generate_src_dst_arrays();

        let mut hd_map = HausdorffMapUndirected::new(&src, &dst, vec![1.0, 2.0, 3.0]);
        hd_map.execute();

        assert!(hd_map.hd_percentile(&1.0).eq(&hd_map.hd()));
        assert!(hd_map.avg_hd_percentile(&0.95).eq(&1.5));
        assert!(hd_map.avg_hd_percentile(&0.0).eq(&0.0));
    }
}