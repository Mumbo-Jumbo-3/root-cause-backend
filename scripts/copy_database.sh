#!/usr/bin/env bash
# Copy one Postgres database into another (Railway -> Neon) and compare row counts.
# Stop writes to the source first: anything written after the dump starts is not copied.
#
#   SOURCE_DATABASE_URL  Railway's DATABASE_PUBLIC_URL
#   TARGET_DATABASE_URL  Neon's direct (unpooled) connection string, pointing at an empty database
set -euo pipefail

: "${SOURCE_DATABASE_URL:?set SOURCE_DATABASE_URL}"
: "${TARGET_DATABASE_URL:?set TARGET_DATABASE_URL}"

# Railway runs Postgres 18; the client tools must be at least that new, whatever is installed locally.
PG_IMAGE="${PG_IMAGE:-postgres:18}"
pg() { docker run --rm -i --network "${DOCKER_NETWORK:-bridge}" "$PG_IMAGE" "$@"; }

dump="$(mktemp -t root-cause-db)"
trap 'rm -f "$dump"' EXIT

echo "Dumping source..."
pg pg_dump --format=custom --no-owner --no-acl --dbname="$SOURCE_DATABASE_URL" > "$dump"

echo "Restoring into target..."
pg pg_restore --no-owner --no-acl --exit-on-error --single-transaction --dbname="$TARGET_DATABASE_URL" < "$dump"

echo "Comparing row counts..."
count_rows() {
  pg psql --no-psqlrc --tuples-only --no-align --dbname="$1" <<'SQL'
SELECT format('select %L || '' '' || count(*) from %I.%I', table_name, table_schema, table_name)
FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
ORDER BY table_name
\gexec
SQL
}
if diff <(count_rows "$SOURCE_DATABASE_URL") <(count_rows "$TARGET_DATABASE_URL"); then
  count_rows "$TARGET_DATABASE_URL"
  echo "Row counts match."
else
  echo "Row counts differ (< source, > target)." >&2
  exit 1
fi
