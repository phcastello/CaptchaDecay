<?php
declare(strict_types=1);

use Random\Engine\Mt19937;
use Random\Randomizer;

const CSV_COLUMNS = [
    'filename', 'label', 'split', 'length', 'width', 'height',
    'perturbation', 'num_lines', 'noise_level', 'font_ratio',
    'image_bg_color', 'text_color', 'line_color', 'seed', 'index',
    'render_seed', 'config_sha256', 'securimage_version',
];

function checked_integer(mixed $value, string $name, int $min, int $max): int
{
    if ((!is_int($value) && !is_string($value)) ||
        !preg_match('/^[0-9]+$/D', (string) $value) ||
        filter_var($value, FILTER_VALIDATE_INT) === false || $value < $min || $value > $max) {
        throw new InvalidArgumentException("$name deve ser inteiro entre $min e $max.");
    }
    return (int) $value;
}

function checked_number(mixed $value, string $name, float $min, float $max): float
{
    if (is_bool($value) || !is_numeric($value) || !is_finite((float) $value) ||
        $value < $min || $value > $max) {
        throw new InvalidArgumentException("$name deve ser número finito entre $min e $max.");
    }
    return (float) $value;
}

function validate_config(array $config): array
{
    foreach (['count' => [1, 1000000], 'seed' => [0, 2147483647],
              'width' => [40, 2000], 'height' => [40, 2000],
              'min_length' => [1, 12], 'max_length' => [1, 12]] as $name => $bounds) {
        $config[$name] = checked_integer($config[$name] ?? null, $name, ...$bounds);
    }
    if ($config['min_length'] > $config['max_length']) {
        throw new InvalidArgumentException('min_length não pode exceder max_length.');
    }
    foreach (['train_ratio', 'validation_ratio'] as $name) {
        $config[$name] = checked_number($config[$name] ?? null, $name, 0, 1);
    }
    if (abs($config['train_ratio'] + $config['validation_ratio'] - 1) > 1e-9) {
        throw new InvalidArgumentException('train_ratio + validation_ratio deve ser 1.');
    }
    if (!is_string($config['output'] ?? null) || trim($config['output']) === '') {
        throw new InvalidArgumentException('output deve ser um caminho não vazio.');
    }
    if (!is_string($config['charset'] ?? null) || !preg_match('/^[a-zA-Z0-9]+$/D', $config['charset']) ||
        count(array_unique(str_split($config['charset']))) !== strlen($config['charset'])) {
        throw new InvalidArgumentException('charset deve conter caracteres ASCII alfanuméricos únicos.');
    }
    foreach (['perturbation' => [0, 1], 'num_lines' => [0, 20],
              'noise_level' => [0, 10], 'font_ratio' => [0.1, 0.99]] as $name => $bounds) {
        $range = $config[$name] ?? null;
        if (!is_array($range) || array_keys($range) !== [0, 1]) {
            throw new InvalidArgumentException("$name deve ser [mínimo, máximo].");
        }
        foreach ($range as $i => $value) {
            $range[$i] = in_array($name, ['num_lines', 'noise_level'], true)
                ? checked_integer($value, $name, (int) $bounds[0], (int) $bounds[1])
                : checked_number($value, $name, ...$bounds);
        }
        if ($range[0] > $range[1]) {
            throw new InvalidArgumentException("$name: mínimo não pode exceder máximo.");
        }
        $config[$name] = $range;
    }
    foreach (['image_bg_color', 'text_color', 'line_color'] as $name) {
        if (!is_array($config[$name] ?? null) || !array_is_list($config[$name]) || !$config[$name]) {
            throw new InvalidArgumentException("$name deve ser uma lista não vazia de cores #RRGGBB.");
        }
        foreach ($config[$name] as $color) {
            if (!is_string($color) || !preg_match('/^#[0-9a-fA-F]{6}$/D', $color)) {
                throw new InvalidArgumentException("$name contém cor inválida; use #RRGGBB.");
            }
        }
    }
    return $config;
}

function parse_options(array $arguments): array
{
    $options = [];
    $allowed = ['count', 'output', 'seed', 'train-ratio', 'validation-ratio', 'config'];
    for ($i = 0; $i < count($arguments); ++$i) {
        if ($arguments[$i] === '--help') {
            return ['help' => true];
        }
        $pair = explode('=', $arguments[$i], 2);
        $name = substr($pair[0], 2);
        if (!str_starts_with($pair[0], '--') || !in_array($name, $allowed, true) || isset($options[$name])) {
            throw new InvalidArgumentException('Opção desconhecida ou repetida: ' . $pair[0]);
        }
        $value = $pair[1] ?? ($arguments[++$i] ?? '');
        if ($value === '' || str_starts_with($value, '--')) {
            throw new InvalidArgumentException("Valor ausente para --$name.");
        }
        $options[$name] = $value;
    }
    $config = require __DIR__ . '/config.php';
    if (isset($options['config'])) {
        if (!is_file($options['config'])) {
            throw new InvalidArgumentException('Arquivo --config não encontrado.');
        }
        $custom = require $options['config'];
        if (!is_array($custom) || array_diff_key($custom, $config)) {
            throw new InvalidArgumentException('--config deve retornar array com chaves conhecidas.');
        }
        $config = array_replace($config, $custom);
        unset($options['config']);
    }
    foreach ($options as $name => $value) {
        $config[str_replace('-', '_', $name)] = $value;
    }
    return validate_config($config);
}

function derived_seed(int $seed, string $key): int
{
    return (int) hexdec(substr(hash('sha256', "$seed:$key"), 0, 7));
}

function sample_metadata(array $config, int $index, string $split): array
{
    // Separate engine: Securimage's mt_rand calls cannot change labels or metadata.
    $rng = new Randomizer(new Mt19937(derived_seed($config['seed'], "sample:$index")));
    $length = $rng->getInt($config['min_length'], $config['max_length']);
    $label = '';
    for ($i = 0; $i < $length; ++$i) {
        $label .= $config['charset'][$rng->getInt(0, strlen($config['charset']) - 1)];
    }
    $sample = ['filename' => sprintf('%08d.png', $index), 'label' => $label,
               'split' => $split, 'length' => $length,
               'width' => $config['width'], 'height' => $config['height']];
    foreach (['perturbation', 'num_lines', 'noise_level', 'font_ratio'] as $name) {
        [$min, $max] = $config[$name];
        $sample[$name] = in_array($name, ['num_lines', 'noise_level'], true)
            ? $rng->getInt($min, $max)
            : round($min + ($max - $min) * $rng->getInt(0, 10000) / 10000, 6);
    }
    foreach (['image_bg_color', 'text_color', 'line_color'] as $name) {
        $sample[$name] = $config[$name][$rng->getInt(0, count($config[$name]) - 1)];
    }
    $visual = array_diff_key($config, array_flip(['count', 'output', 'seed', 'train_ratio', 'validation_ratio']));
    $sample += ['seed' => $config['seed'], 'index' => $index,
                'render_seed' => $rng->getInt(0, 2147483647),
                'config_sha256' => hash('sha256', json_encode($visual, JSON_THROW_ON_ERROR)),
                'securimage_version' => '4.0.2'];
    return $sample;
}

function batch_splits(array $config, int $start): array
{
    $train = (int) round($config['count'] * $config['train_ratio']);
    $splits = array_merge(array_fill(0, $train, 'train'),
                         array_fill(0, $config['count'] - $train, 'validation'));
    $rng = new Randomizer(new Mt19937(derived_seed($config['seed'], "splits:$start")));
    return $rng->shuffleArray($splits);
}

function render_png(array $sample): string
{
    $level = error_reporting();
    // Legacy upstream emits deprecations on recent PHP; do not change vendor code.
    error_reporting($level & ~E_DEPRECATED);
    ob_start();
    try {
        $image = new Securimage([
            'display_value' => $sample['label'], 'image_type' => Securimage::SI_IMAGE_PNG,
            'image_width' => $sample['width'], 'image_height' => $sample['height'],
            'perturbation' => $sample['perturbation'], 'num_lines' => $sample['num_lines'],
            'noise_level' => $sample['noise_level'], 'font_ratio' => $sample['font_ratio'],
            'image_bg_color' => $sample['image_bg_color'], 'text_color' => $sample['text_color'],
            'line_color' => $sample['line_color'],
            'no_session' => true, 'no_exit' => true, 'send_headers' => false,
        ]);
        mt_srand($sample['render_seed'], MT_RAND_MT19937);
        try {
            $image->show();
        } catch (Throwable $error) {
            restore_error_handler();
            throw $error;
        }
        return (string) ob_get_contents();
    } finally {
        ob_end_clean();
        error_reporting($level);
    }
}

function validate_png(string $data, int $width, int $height): void
{
    $info = getimagesizefromstring($data);
    if (!str_starts_with($data, "\x89PNG\r\n\x1a\n") || !$info ||
        $info[2] !== IMAGETYPE_PNG || $info[0] !== $width || $info[1] !== $height) {
        throw new RuntimeException('Gerador não produziu PNG válido nas dimensões configuradas.');
    }
}

function write_all($stream, string $data): void
{
    for ($offset = 0; $offset < strlen($data); $offset += $written) {
        $written = fwrite($stream, substr($data, $offset));
        if ($written === false || $written === 0) {
            throw new RuntimeException('Falha ao escrever arquivo; progresso anterior preservado.');
        }
    }
}

function generate_dataset(array $config): array
{
    $config = validate_config($config);
    $root = rtrim($config['output'], '/\\');
    foreach (['train', 'validation'] as $split) {
        if (!is_dir("$root/$split") && !mkdir("$root/$split", 0777, true)) {
            throw new RuntimeException("Não foi possível criar $root/$split.");
        }
    }
    $lock = fopen("$root/.generation.lock", 'c');
    if (!$lock || !flock($lock, LOCK_EX | LOCK_NB)) {
        throw new RuntimeException('Outra geração está usando este diretório.');
    }
    $csv = null;
    try {
        $csv = fopen("$root/labels.csv", 'c+b');
        if (!$csv) {
            throw new RuntimeException('Não foi possível abrir labels.csv.');
        }
        $files = [];
        $indices = [];
        $header = fgetcsv($csv, 0, ',', '"', '');
        if ($header !== false && $header !== CSV_COLUMNS) {
            throw new RuntimeException('Cabeçalho incompatível em labels.csv; arquivo preservado.');
        }
        while (($row = fgetcsv($csv, 0, ',', '"', '')) !== false) {
            if (count($row) !== count(CSV_COLUMNS)) {
                throw new RuntimeException('Linha incompleta em labels.csv; revise antes de continuar.');
            }
            $record = array_combine(CSV_COLUMNS, $row);
            $file = $record['filename'];
            if (!preg_match('/^[0-9]{8}\.png$/D', $file) ||
                !in_array($record['split'], ['train', 'validation'], true) ||
                isset($indices[$file]) || !is_file("$root/{$record['split']}/$file") || $record['label'] === '') {
                throw new RuntimeException('Referência ausente ou duplicada em labels.csv; arquivo preservado.');
            }
            $indices[$file] = (int) substr($file, 0, 8);
            $files["{$record['split']}/$file"] = true;
        }
        $size = ftell($csv);
        if ($size > 0) {
            fseek($csv, -1, SEEK_END);
            if (fread($csv, 1) !== "\n") {
                throw new RuntimeException('labels.csv foi interrompido durante escrita; revise a última linha.');
            }
        }
        foreach (['train', 'validation'] as $split) {
            foreach (glob("$root/$split/*.png") as $path) {
                if (!isset($files[$split . '/' . basename($path)])) {
                    throw new RuntimeException('PNG sem metadata em labels.csv; preserve e revise o arquivo antes de continuar.');
                }
            }
        }
        fseek($csv, 0, SEEK_END);
        if ($header === false && fputcsv($csv, CSV_COLUMNS, ',', '"', '') === false) {
            throw new RuntimeException('Falha ao escrever cabeçalho CSV.');
        }
        $start = $indices ? max($indices) + 1 : 1;
        if ($start + $config['count'] - 1 > 99999999) {
            throw new RuntimeException('Sequência de nomes esgotada.');
        }
        $saved = ['train' => 0, 'validation' => 0];
        foreach (batch_splits($config, $start) as $offset => $split) {
            $sample = sample_metadata($config, $start + $offset, $split);
            $data = render_png($sample);
            validate_png($data, $config['width'], $config['height']);
            $path = "$root/$split/{$sample['filename']}";
            $output = fopen($path, 'xb');
            if (!$output) {
                throw new RuntimeException("Não foi possível criar $path sem sobrescrever.");
            }
            try {
                write_all($output, $data);
                if (!fflush($output)) {
                    throw new RuntimeException("Falha ao salvar $path.");
                }
            } finally {
                fclose($output);
            }
            $position = ftell($csv);
            if (fputcsv($csv, array_values($sample), ',', '"', '') === false || !fflush($csv)) {
                ftruncate($csv, $position);
                throw new RuntimeException('Falha no CSV; imagem preservada sem metadata. Revise antes de continuar.');
            }
            ++$saved[$split];
        }
        return $saved;
    } finally {
        if (is_resource($csv)) {
            fclose($csv);
        }
        flock($lock, LOCK_UN);
        fclose($lock);
    }
}

function main(array $arguments): int
{
    ini_set('display_errors', 'stderr');
    try {
        $config = parse_options($arguments);
        if (isset($config['help'])) {
            echo "Uso: php generator/generate.php [--count 1000] [--output dataset/synthetic] " .
                 "[--seed 42] [--train-ratio 0.9] [--validation-ratio 0.1] [--config arquivo.php]\n";
            return 0;
        }
        $autoload = __DIR__ . '/../vendor/autoload.php';
        if (!is_file($autoload)) {
            throw new RuntimeException('Dependência ausente; execute composer install.');
        }
        require_once $autoload;
        if (!extension_loaded('gd') || !(gd_info()['FreeType Support'] ?? false) ||
            !is_readable(Securimage::getPath() . '/AHGBold.ttf')) {
            throw new RuntimeException('São necessários GD, FreeType e a fonte fornecida pelo Securimage.');
        }
        $saved = generate_dataset($config);
        echo "Concluído: train={$saved['train']}, validation={$saved['validation']}.\n";
        return 0;
    } catch (Throwable $error) {
        fwrite(STDERR, "Erro: {$error->getMessage()}\n");
        return 1;
    }
}

if (realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    exit(main(array_slice($argv, 1)));
}
