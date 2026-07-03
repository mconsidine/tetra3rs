use crate::error::{Error, Result};
use std::path::Path;

pub struct GaiaStar {
    pub source_id: i64,
    pub ra_deg: f32,
    pub dec_deg: f32,
    pub phot_g_mean_mag: f32,
    pub pmra: Option<f32>,
    pub pmdec: Option<f32>,
}

/// Load a Gaia catalog from the binary format.
///
/// Binary format spec:
/// - Header: magic "GDR3" (4 bytes) + version (u32 LE, value 1) + num_stars (u64 LE)
/// - Per star (36 bytes): source_id (i64 LE) + ra (f64 LE) + dec (f64 LE) + mag (f32 LE) + pmra (f32 LE) + pmdec (f32 LE)
pub fn load_gaia_binary<P: AsRef<Path>>(path: P) -> Result<Vec<GaiaStar>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;

    // Validate magic
    if &header[0..4] != b"GDR3" {
        return Err(Error::InvalidCatalog(
            "Gaia binary: expected magic 'GDR3'".into(),
        ));
    }

    // Validate version
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != 1 {
        return Err(Error::InvalidCatalog(format!(
            "Gaia binary: unsupported version {}, expected 1",
            version
        )));
    }

    let num_stars = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;

    let record_size = 36;
    // A corrupt/truncated header could claim a huge `num_stars`, driving a
    // multi-gigabyte allocation (or a capacity-overflow abort) before
    // `read_exact` gets a chance to fail. Require the star block to match the
    // file's remaining bytes exactly first.
    let expected_bytes = num_stars.checked_mul(record_size).ok_or_else(|| {
        Error::InvalidCatalog("Gaia binary: num_stars * record_size overflows".into())
    })?;
    let data_bytes = file.metadata()?.len().saturating_sub(header.len() as u64);
    if expected_bytes as u64 != data_bytes {
        return Err(Error::InvalidCatalog(format!(
            "Gaia binary: header claims {num_stars} stars ({expected_bytes} bytes) \
             but file has {data_bytes} data bytes"
        )));
    }
    let mut buf = vec![0u8; expected_bytes];
    file.read_exact(&mut buf)?;

    let mut stars = Vec::with_capacity(num_stars);
    for i in 0..num_stars {
        let offset = i * record_size;
        let rec = &buf[offset..offset + record_size];

        let source_id = i64::from_le_bytes(rec[0..8].try_into().unwrap());
        let ra = f64::from_le_bytes(rec[8..16].try_into().unwrap());
        let dec = f64::from_le_bytes(rec[16..24].try_into().unwrap());
        let mag = f32::from_le_bytes(rec[24..28].try_into().unwrap());
        let pmra = f32::from_le_bytes(rec[28..32].try_into().unwrap());
        let pmdec = f32::from_le_bytes(rec[32..36].try_into().unwrap());

        stars.push(GaiaStar {
            source_id,
            ra_deg: ra as f32,
            dec_deg: dec as f32,
            phot_g_mean_mag: mag,
            pmra: Some(pmra),
            pmdec: Some(pmdec),
        });
    }

    Ok(stars)
}
