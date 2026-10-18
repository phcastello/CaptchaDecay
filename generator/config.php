<?php
// Initial calibration from 63 confirmed real/train labels and manual train-only inspection.
// These are revisable hypotheses, not measured parameters of the remote generator.
return [
    'count' => 1000,
    'output' => __DIR__ . '/../dataset/synthetic',
    'seed' => 42,
    'train_ratio' => 0.9,
    'validation_ratio' => 0.1,
    'charset' => 'abcdefghiklmnprstuvwyz23456789',
    'width' => 215,
    'height' => 80,
    'min_length' => 4,
    'max_length' => 6,
    'length_distribution' => [4 => 27, 5 => 46, 6 => 27], // Relative integer weights; null selects uniform.
    'perturbation' => [0.90, 1.0],
    'num_lines' => [5, 8],
    'noise_level' => [2, 4],
    'font_ratio' => [0.42, 0.52],
    'image_bg_color' => ['#ffffff', '#f6f6f6'],
    'text_color' => ['#606060', '#707070', '#808080'],
    'line_color' => ['#606060', '#808080'],
];
