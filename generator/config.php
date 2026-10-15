<?php
// Initial ranges follow Securimage capabilities, not measured Correios parameters.
// TODO: Calibrate charset, length distribution and visual ranges after manual labeling.
return [
    'count' => 1000,
    'output' => __DIR__ . '/../dataset/synthetic',
    'seed' => 42,
    'train_ratio' => 0.9,
    'validation_ratio' => 0.1,
    'charset' => 'abcdefghijkmnopqrstuvwxzyABCDEFGHJKLMNPQRSTUVWXZY0123456789',
    'width' => 215,
    'height' => 80,
    'min_length' => 4,
    'max_length' => 6,
    'perturbation' => [0.65, 1.0],
    'num_lines' => [3, 7],
    'noise_level' => [1, 4],
    'font_ratio' => [0.35, 0.45],
    'image_bg_color' => ['#ffffff', '#f6f6f6'],
    'text_color' => ['#606060', '#707070', '#808080'],
    'line_color' => ['#707070', '#909090'],
];
