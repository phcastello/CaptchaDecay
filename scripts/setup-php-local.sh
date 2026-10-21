#!/usr/bin/env bash
# Rootless PHP/Composer setup using packages from Ubuntu's configured apt sources.
# Packages are extracted only; maintainer scripts and system services are not run.
set -euo pipefail
captcha_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 24.04 || "$(dpkg --print-architecture)" != amd64 ]]; then
    printf '%s\n' 'Este setup local requer Ubuntu 24.04 amd64.' >&2
    exit 1
fi
for captcha_command in apt-get dpkg-deb curl sha384sum; do
    command -v "$captcha_command" >/dev/null
done
mkdir -p -- "$captcha_root/.tools/php-debs" "$captcha_root/.tools/php"
captcha_php_complete=true
[[ -x "$captcha_root/.tools/php/usr/bin/php8.3" ]] || captcha_php_complete=false
for captcha_extension in ctype fileinfo iconv mbstring phar tokenizer dom \
        simplexml xml xmlreader xmlwriter curl gd; do
    [[ -f "$captcha_root/.tools/php/usr/lib/php/20230831/$captcha_extension.so" ]] \
        || captcha_php_complete=false
done
if [[ "$captcha_php_complete" != true ]]; then
    (
        cd -- "$captcha_root/.tools/php-debs"
        apt-get download php8.3-cli php8.3-common php8.3-gd \
            php8.3-mbstring php8.3-xml php8.3-curl
        for captcha_deb in php8.3-{cli,common,gd,mbstring,xml,curl}_*.deb; do
            dpkg-deb --extract "$captcha_deb" "$captcha_root/.tools/php"
        done
    )
fi
"$captcha_root/scripts/php-local" -r '
    if (!extension_loaded("gd") || !gd_info()["FreeType Support"]) {
        fwrite(STDERR, "GD/FreeType indisponível.\n"); exit(1);
    }
    echo PHP_VERSION, " — GD/FreeType OK\n";
'
if [[ ! -f "$captcha_root/.tools/composer.phar" ]]; then
    captcha_installer="$(mktemp "$captcha_root/.tools/composer-installer.XXXXXXXX.php")"
    trap '[[ ! -f "$captcha_installer" ]] || unlink -- "$captcha_installer"' EXIT
    curl --fail --silent --show-error --location \
        https://getcomposer.org/installer --output "$captcha_installer"
    captcha_expected="$(curl --fail --silent --show-error --location https://composer.github.io/installer.sig)"
    captcha_actual="$(sha384sum "$captcha_installer")"
    captcha_actual="${captcha_actual%% *}"
    if [[ "$captcha_expected" != "$captcha_actual" ]]; then
        printf '%s\n' 'Checksum SHA-384 do instalador Composer inválido.' >&2
        exit 1
    fi
    "$captcha_root/scripts/php-local" "$captcha_installer" \
        --install-dir="$captcha_root/.tools" --filename=composer.phar
fi
"$captcha_root/scripts/composer-local" --version
