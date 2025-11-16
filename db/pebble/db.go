package pebble

import (
	"bytes"
	"context"
	"fmt"
	"os"

	"github.com/cockroachdb/pebble"
	"github.com/magiconair/properties"
	"github.com/pingcap/go-ycsb/pkg/ycsb"
)

type pebbleDB struct {
	db *pebble.DB
}

type pebbleCreator struct{}

func init() {
	fmt.Println(">>> registering my pebble wrapper... <<<")
	ycsb.RegisterDBCreator("pebble", &pebbleCreator{})
}

func (c *pebbleCreator) Create(p *properties.Properties) (ycsb.DB, error) {
	dir := p.GetString("pebble.dir", "/tmp/pebble")
	drop := p.GetBool("dropdata", false)

	if drop {
		if err := os.RemoveAll(dir); err != nil {
			return nil, fmt.Errorf("pebble: dropdata failed: %w", err)
		}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("pebble: mkdir %s: %w", dir, err)
	}

	opts := &pebble.Options{}
	db, err := pebble.Open(dir, opts)
	if err != nil {
		return nil, fmt.Errorf("pebble: open: %w", err)
	}

	return &pebbleDB{db: db}, nil
}

func (p *pebbleDB) Close() error {
	return p.db.Close()
}

func (p *pebbleDB) InitThread(ctx context.Context, threadID int, threadCount int) context.Context {
	return ctx
}

func (p *pebbleDB) CleanupThread(ctx context.Context) {

}

func encodeKey(table, key string) []byte {
	return []byte(table + ":" + key)
}

func (pdb *pebbleDB) Read(ctx context.Context, table string, key string, fields []string) (map[string][]byte, error) {
	k := encodeKey(table, key)
	val, closer, err := pdb.db.Get(k)
	if err != nil {
		if err == pebble.ErrNotFound {
			return nil, nil
		}
		return nil, err
	}
	defer closer.Close()

	data := append([]byte(nil), val...)
	return decodeValues(data, fields), nil
}

func (p *pebbleDB) Scan(ctx context.Context, table string, startKey string, count int, fields []string) ([]map[string][]byte, error) {
    prefix := []byte(table + ":")
    start := encodeKey(table, startKey)

    iter, err := p.db.NewIter(&pebble.IterOptions{
        LowerBound: start,
    })
    if err != nil {
        return nil, err
    }
    defer iter.Close()

    var res []map[string][]byte

    // start from the first key at or after LowerBound
    for ok := iter.First(); ok && len(res) < count; ok = iter.Next() {
        k := iter.Key()
        if !bytes.HasPrefix(k, prefix) {
            break
        }
        v := iter.Value()
        data := append([]byte(nil), v...) // copy out of pebble's buffer
        res = append(res, decodeValues(data, fields))
    }

    return res, nil
}

func (p *pebbleDB) Update(ctx context.Context, table string, key string, values map[string][]byte) error {
	// YCSB semantics: update is overwrite.
	return p.Insert(ctx, table, key, values)
}

func (p *pebbleDB) Insert(ctx context.Context, table string, key string, values map[string][]byte) error {
	k := encodeKey(table, key)
	data := encodeValues(values)
	return p.db.Set(k, data, pebble.Sync)
}

func (p *pebbleDB) Delete(ctx context.Context, table string, key string) error {
	k := encodeKey(table, key)
	return p.db.Delete(k, pebble.Sync)
}

func encodeValues(values map[string][]byte) []byte {
	if len(values) == 0 {
		return nil
	}

	var buf bytes.Buffer
	first := true
	for k, v := range values {
		if !first {
			buf.WriteByte('\n')
		}
		first = false
		buf.WriteString(k)
		buf.WriteByte('=')
		buf.Write(v)
	}
	return buf.Bytes()
}

func decodeValues(data []byte, fields []string) map[string][]byte {
	res := make(map[string][]byte)
	if len(data) == 0 {
		return res
	}

	var want map[string]struct{}
	if len(fields) > 0 {
		want = make(map[string]struct{}, len(fields))
		for _, f := range fields {
			want[f] = struct{}{}
		}
	}

	for len(data) > 0 {
		eq := bytes.IndexByte(data, '=')
		if eq <= 0 {
			break
		}
		key := string(data[:eq])
		data = data[eq+1:]

		nl := bytes.IndexByte(data, '\n')
		var val []byte
		if nl < 0 {
			val = data
			data = nil
		} else {
			val = data[:nl]
			data = data[nl+1:]
		}

		if want == nil {
			res[key] = append([]byte(nil), val...)
		} else {
			if _, ok := want[key]; ok {
				res[key] = append([]byte(nil), val...)
			}
		}
	}

	return res
}
