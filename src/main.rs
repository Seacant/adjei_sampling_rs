extern crate clap;
extern crate serde;
extern crate statistical;
extern crate statrs;

use clap::{App, Arg};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use statistical::{mean, standard_deviation};
use statrs::distribution::{Continuous, StudentsT};
use std::cmp::Ordering;
use std::fs::File;

#[derive(Debug, Deserialize, Clone)]
struct Instance {
  condition: String,
  mid: f64,
  pre: f64,
  gain: f64,

  #[serde(rename = "final")]
  post: f64,
}

#[derive(Debug)]
struct Match {
  big: Instance,
  small: Instance,
}

#[derive(Debug, Serialize, Clone)]
struct Output {
  small_pre_mean: f64,
  small_post_mean: f64,
  small_mid_mean: f64,
  small_gain_mean: f64,

  big_pre_mean: f64,
  big_post_mean: f64,
  big_mid_mean: f64,
  big_gain_mean: f64,

  small_pre_stdev: f64,
  small_post_stdev: f64,
  small_mid_stdev: f64,
  small_gain_stdev: f64,

  big_pre_stdev: f64,
  big_post_stdev: f64,
  big_mid_stdev: f64,
  big_gain_stdev: f64,

  post_t_pvalue: f64,
  post_t_tvalue: f64,
}

fn read_csv_data(filename: &str) -> Box<Vec<Instance>> {
  let file: File = File::open(filename).unwrap();
  let mut reader = csv::Reader::from_reader(file);

  let mut data: Vec<Instance> = Vec::new();
  for datum in reader.deserialize() {
    let instance: Instance = datum.unwrap();
    data.push(instance);
  }

  return Box::new(data);
}

fn main() {
  // Declare cli args
  let opts = App::new("Data sample statistics tester")
    .version("0.1.0")
    .author("Travis Fletcher (Seacant)")
    .about("Sample and run statistics on the sample data")
    .arg(
      Arg::with_name("iterations")
        .short("i")
        .long("iterations")
        .value_name("N")
        .help("Number of iterations to run")
        .takes_value(true),
    )
    .arg(
      Arg::with_name("input")
        .index(1)
        .required(true)
        .help("Filename to run statistics on"),
    )
    .get_matches();

  // Read in CSV data as specified by input parameter
  let mut data = *read_csv_data(opts.value_of("input").unwrap());

  // Separate Big-Group and Small-Group-To-Match
  let (all_big_boys, all_small_boys): (Vec<Instance>, Vec<Instance>) = data
    .drain(..)
    .partition(|element| element.condition == "Big-Group");

  let mut outputs: Vec<Output> = Vec::with_capacity(
    opts
      .value_of("iterations")
      .unwrap()
      .parse::<usize>()
      .unwrap(),
  );
  let mut rng = thread_rng();

  // Do as many iterations as specified in argument
  for _ in 0..(opts.value_of("iterations").unwrap().parse::<i32>().unwrap()) {
    // Clone our lists so we can pop off safely
    let mut big_boys = all_big_boys.clone();
    let mut small_boys = all_small_boys.clone();

    let mut matches: Vec<Match> = Vec::new();

    // Shuffle small_boys
    small_boys.shuffle(&mut rng);

    for record in small_boys.drain(..) {
      // Sort big_boys by the absolute value of the difference between it's post
      // and the current smallboy's record's post.
      // Should be O(nlog(n)). Phew.
      big_boys.sort_by(|a, b| {
        (a.pre - record.pre)
          .abs()
          .partial_cmp(&(b.pre - record.pre).abs())
          .unwrap_or(Ordering::Equal)
          .reverse()
      });

      // Because of the above sort, this pop returns the closest value in O(1).
      let matched_record = big_boys.pop().unwrap();

      matches.push(Match {
        small: record,
        big: matched_record,
      });
    }

    let t_test_result = paired_t(
      matches.iter().map(|e| e.small.post).collect::<Vec<f64>>(),
      matches.iter().map(|e| e.big.post).collect::<Vec<f64>>(),
    );

    let output = Output {
      small_pre_mean: mean(&matches.iter().map(|e| e.small.pre).collect::<Vec<f64>>()[..]),
      small_post_mean: mean(&matches.iter().map(|e| e.small.post).collect::<Vec<f64>>()[..]),
      small_mid_mean: mean(&matches.iter().map(|e| e.small.mid).collect::<Vec<f64>>()[..]),
      small_gain_mean: mean(&matches.iter().map(|e| e.small.gain).collect::<Vec<f64>>()[..]),

      big_pre_mean: mean(&matches.iter().map(|e| e.big.pre).collect::<Vec<f64>>()[..]),
      big_post_mean: mean(&matches.iter().map(|e| e.big.post).collect::<Vec<f64>>()[..]),
      big_mid_mean: mean(&matches.iter().map(|e| e.big.mid).collect::<Vec<f64>>()[..]),
      big_gain_mean: mean(&matches.iter().map(|e| e.big.gain).collect::<Vec<f64>>()[..]),

      small_pre_stdev: standard_deviation(
        &matches.iter().map(|e| e.small.pre).collect::<Vec<f64>>()[..],
        None,
      ),
      small_post_stdev: standard_deviation(
        &matches.iter().map(|e| e.small.post).collect::<Vec<f64>>()[..],
        None,
      ),
      small_mid_stdev: standard_deviation(
        &matches.iter().map(|e| e.small.mid).collect::<Vec<f64>>()[..],
        None,
      ),
      small_gain_stdev: standard_deviation(
        &matches.iter().map(|e| e.small.gain).collect::<Vec<f64>>()[..],
        None,
      ),

      big_pre_stdev: standard_deviation(
        &matches.iter().map(|e| e.big.pre).collect::<Vec<f64>>()[..],
        None,
      ),
      big_post_stdev: standard_deviation(
        &matches.iter().map(|e| e.big.post).collect::<Vec<f64>>()[..],
        None,
      ),
      big_mid_stdev: standard_deviation(
        &matches.iter().map(|e| e.big.mid).collect::<Vec<f64>>()[..],
        None,
      ),
      big_gain_stdev: standard_deviation(
        &matches.iter().map(|e| e.big.gain).collect::<Vec<f64>>()[..],
        None,
      ),

      post_t_pvalue: t_test_result.p,
      post_t_tvalue: t_test_result.t,
    };

    outputs.push(output);
  }

  // Save the iterations
  let mut writer = csv::Writer::from_path("iterations.csv").unwrap();
  for record in outputs.clone() {
    writer.serialize(record).unwrap();
  }

  let small_pre_mean_mean = mean(
    &outputs
      .iter()
      .map(|e| e.small_pre_mean)
      .collect::<Vec<f64>>()[..],
  );
  let big_pre_mean_mean = mean(&outputs.iter().map(|e| e.big_pre_mean).collect::<Vec<f64>>()[..]);
  let small_post_mean_mean = mean(
    &outputs
      .iter()
      .map(|e| e.small_post_mean)
      .collect::<Vec<f64>>()[..],
  );
  let big_post_mean_mean = mean(
    &outputs
      .iter()
      .map(|e| e.big_post_mean)
      .collect::<Vec<f64>>()[..],
  );
  let small_mid_mean_mean = mean(
    &outputs
      .iter()
      .map(|e| e.small_mid_mean)
      .collect::<Vec<f64>>()[..],
  );
  let big_mid_mean_mean = mean(&outputs.iter().map(|e| e.big_mid_mean).collect::<Vec<f64>>()[..]);
  let small_pre_stdev_mean = mean(
    &outputs
      .iter()
      .map(|e| e.small_pre_stdev)
      .collect::<Vec<f64>>()[..],
  );
  let big_pre_stdev_mean = mean(
    &outputs
      .iter()
      .map(|e| e.big_pre_stdev)
      .collect::<Vec<f64>>()[..],
  );
  let small_post_stdev_mean = mean(
    &outputs
      .iter()
      .map(|e| e.small_post_stdev)
      .collect::<Vec<f64>>()[..],
  );
  let big_post_stdev_mean = mean(
    &outputs
      .iter()
      .map(|e| e.big_post_stdev)
      .collect::<Vec<f64>>()[..],
  );
  let small_mid_stdev_mean = mean(
    &outputs
      .iter()
      .map(|e| e.small_mid_stdev)
      .collect::<Vec<f64>>()[..],
  );
  let big_mid_stdev_mean = mean(
    &outputs
      .iter()
      .map(|e| e.big_mid_stdev)
      .collect::<Vec<f64>>()[..],
  );

  let post_t_pvalue_mean = mean(
    &outputs
      .iter()
      .map(|e| e.post_t_pvalue)
      .collect::<Vec<f64>>()[..],
  );
  let post_t_tvalue_mean = mean(
    &outputs
      .iter()
      .map(|e| e.post_t_tvalue)
      .collect::<Vec<f64>>()[..],
  );

  let small_pre_mean_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.small_pre_mean)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let big_pre_mean_stdev = standard_deviation(
    &outputs.iter().map(|e| e.big_pre_mean).collect::<Vec<f64>>()[..],
    None,
  );
  let small_post_mean_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.small_post_mean)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let big_post_mean_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.big_post_mean)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let small_mid_mean_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.small_mid_mean)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let big_mid_mean_stdev = standard_deviation(
    &outputs.iter().map(|e| e.big_mid_mean).collect::<Vec<f64>>()[..],
    None,
  );

  let small_pre_stdev_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.small_pre_stdev)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let big_pre_stdev_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.big_pre_stdev)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let small_post_stdev_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.small_post_stdev)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let big_post_stdev_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.big_post_stdev)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let small_mid_stdev_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.small_mid_stdev)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let big_mid_stdev_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.big_mid_stdev)
      .collect::<Vec<f64>>()[..],
    None,
  );

  let post_t_pvalue_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.post_t_pvalue)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let post_t_tvalue_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.post_t_tvalue)
      .collect::<Vec<f64>>()[..],
    None,
  );

  let small_gain_mean_mean = mean(
    &outputs
      .iter()
      .map(|e| e.small_gain_mean)
      .collect::<Vec<f64>>()[..],
  );
  let big_gain_mean_mean = mean(
    &outputs
      .iter()
      .map(|e| e.big_gain_mean)
      .collect::<Vec<f64>>()[..],
  );
  let small_gain_mean_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.small_gain_mean)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let big_gain_mean_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.big_gain_mean)
      .collect::<Vec<f64>>()[..],
    None,
  );

  let small_gain_stdev_mean = mean(
    &outputs
      .iter()
      .map(|e| e.small_gain_stdev)
      .collect::<Vec<f64>>()[..],
  );
  let big_gain_stdev_mean = mean(
    &outputs
      .iter()
      .map(|e| e.big_gain_stdev)
      .collect::<Vec<f64>>()[..],
  );
  let small_gain_stdev_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.small_gain_stdev)
      .collect::<Vec<f64>>()[..],
    None,
  );
  let big_gain_stdev_stdev = standard_deviation(
    &outputs
      .iter()
      .map(|e| e.big_gain_stdev)
      .collect::<Vec<f64>>()[..],
    None,
  );

  let proportion_significant =
    outputs.iter().filter(|e| e.post_t_pvalue < 0.05).count() as f64 / outputs.len() as f64;

  println!("small_pre_mean_mean = {}", small_pre_mean_mean);
  println!("big_pre_mean_mean = {}", big_pre_mean_mean);
  println!("small_post_mean_mean = {}", small_post_mean_mean);
  println!("big_post_mean_mean = {}", big_post_mean_mean);
  println!("small_mid_mean_mean = {}", small_mid_mean_mean);
  println!("big_mid_mean_mean = {}", big_mid_mean_mean);
  println!("small_pre_stdev_mean = {}", small_pre_stdev_mean);
  println!("big_pre_stdev_mean = {}", big_pre_stdev_mean);
  println!("small_post_stdev_mean = {}", small_post_stdev_mean);
  println!("big_post_stdev_mean = {}", big_post_stdev_mean);
  println!("small_mid_stdev_mean = {}", small_mid_stdev_mean);
  println!("big_mid_stdev_mean = {}", big_mid_stdev_mean);
  println!("post_t_pvalue_mean = {}", post_t_pvalue_mean);
  println!("post_t_tvalue_mean = {}", post_t_tvalue_mean);
  println!("small_pre_mean_stdev = {}", small_pre_mean_stdev);
  println!("big_pre_mean_stdev = {}", big_pre_mean_stdev);
  println!("small_post_mean_stdev = {}", small_post_mean_stdev);
  println!("big_post_mean_stdev = {}", big_post_mean_stdev);
  println!("small_mid_mean_stdev = {}", small_mid_mean_stdev);
  println!("big_mid_mean_stdev = {}", big_mid_mean_stdev);
  println!("small_pre_stdev_stdev = {}", small_pre_stdev_stdev);
  println!("big_pre_stdev_stdev = {}", big_pre_stdev_stdev);
  println!("small_post_stdev_stdev = {}", small_post_stdev_stdev);
  println!("big_post_stdev_stdev = {}", big_post_stdev_stdev);
  println!("small_mid_stdev_stdev = {}", small_mid_stdev_stdev);
  println!("big_mid_stdev_stdev = {}", big_mid_stdev_stdev);
  println!("small_gain_mean_mean = {}", small_gain_mean_mean);
  println!("small_gain_mean_stdev = {}", small_gain_mean_stdev);
  println!("big_gain_mean_mean = {}", big_gain_mean_mean);
  println!("big_gain_mean_stdev = {}", big_gain_mean_stdev);
  println!("small_gain_stdev_mean = {}", small_gain_stdev_mean);
  println!("small_gain_stdev_stdev = {}", small_gain_stdev_stdev);
  println!("big_gain_stdev_mean = {}", big_gain_stdev_mean);
  println!("big_gain_stdev_stdev = {}", big_gain_stdev_stdev);
  println!("post_t_pvalue_mean = {}", post_t_pvalue_mean);
  println!("post_t_pvalue_stdev = {}", post_t_pvalue_stdev);
  println!("post_t_tvalue_mean = {}", post_t_tvalue_mean);
  println!("post_t_tvalue_stdev = {}", post_t_tvalue_stdev);
  println!("proportion_significant = {}", proportion_significant);
}

struct TTestResult {
  p: f64,
  t: f64,
}
fn paired_t(a: Vec<f64>, b: Vec<f64>) -> TTestResult {
  let n = a.len();

  let d = a
    .iter()
    .zip(b.iter())
    .map(|(a, b)| a - b)
    .collect::<Vec<f64>>();
  let dbar = mean(&d[..]);
  let sd = standard_deviation(&d[..], None);

  let se_dbar = sd / (n as f64).sqrt();

  let t = dbar / se_dbar;

  let t_tester = StudentsT::new(0.0, 1.0, (n - 1) as f64).unwrap();
  let p = t_tester.pdf(t);

  return TTestResult { p, t };
}
