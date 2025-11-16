package pebblelru

import (
	"bytes"
	"context"
	"fmt"

	lrucache "github.com/rohannunu/pebble-cs598rap/lru-cache"

	"github.com/magiconair/properties"
	"github.com/pingcap/go-ycsb/pkg/ycsb"
)

// pebbleLRUDB implements ycsb.DB using your LRU cache (which wraps Pebble).
type pebbleLRUDB struct {
	lru *lrucache.LRUCache
}

type pebbleLRUCreator struct{}

func init() {
	ycsb.RegisterDBCreator("pebblelru", &pebbleLRUCreator{})
}

// Create initializes the LRU cache DB.
// You can configure the capacity with: -p pebble.lru_capacity=...
func (c *pebbleLRUCreator) Create(p *properties.Properties) (ycsb.DB, error) {
	fmt.Println(">>> registering pebble LRU wrapper... <<<")
	capacity := p.GetInt("pebble.lru_capacity", 1024)
	lru := lrucache.NewLRUCache(capacity)
	return &pebbleLRUDB{lru: lru}, nil
}

func (db *pebbleLRUDB) Close() error {
	// If you later add a Close() method to LRUCache or the underlying cache,
	// call it here. For now, nothing to do.
	return nil
}

func (db *pebbleLRUDB) InitThread(ctx context.Context, threadID int, threadCount int) context.Context {
	return ctx
}

func (db *pebbleLRUDB) CleanupThread(ctx context.Context) {}

// ---------- key/value helpers ----------

func encodeKey(table, key string) []byte {
	return []byte(table + ":" + key)
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

// ---------- YCSB operations ----------

func (db *pebbleLRUDB) Read(ctx context.Context, table string, key string, fields []string) (map[string][]byte, error) {
	k := encodeKey(table, key)

	val, found, err := db.lru.Get(k)
	if err != nil {
		return nil, err
	}
	if !found {
		// YCSB convention: not found -> (nil, nil)
		return nil, nil
	}

	// Copy value out before decoding
	data := append([]byte(nil), val...)
	return decodeValues(data, fields), nil
}

func (db *pebbleLRUDB) Scan(ctx context.Context, table string, startKey string, count int, fields []string) ([]map[string][]byte, error) {
	// Your LRU wrapper doesn't expose a scan over the keyspace.
	// Most YCSB core workloads set scanproportion=0, so we can
	// safely return empty here for now.
	//
	// If you later want to support scan, you can plumb through a
	// Pebble iterator from the underlying cache/Pebble.
	return nil, nil
}

func (db *pebbleLRUDB) Update(ctx context.Context, table string, key string, values map[string][]byte) error {
	// YCSB semantics: update is just overwrite of the whole record.
	return db.Insert(ctx, table, key, values)
}

func (db *pebbleLRUDB) Insert(ctx context.Context, table string, key string, values map[string][]byte) error {
	k := encodeKey(table, key)
	data := encodeValues(values)

	_, err := db.lru.Set(k, data, true /* to_cache */)
	return err
}

func (db *pebbleLRUDB) Delete(ctx context.Context, table string, key string) error {
	// Your LRUCache API doesn’t expose a Delete directly.
	// Simplest is to evict from cache + let underlying Pebble
	// delete if you add that behavior in your Cache layer.
	// For now, we treat Delete as a no-op (most YCSB core workloads
	// don’t use delete anyway).
	return nil
}
