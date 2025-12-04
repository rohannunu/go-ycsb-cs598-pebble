package pebbledetox

import (
	"bytes"
	"context"
	"fmt"
	"sync"

	detoxcache "github.com/rohannunu/pebble-cs598rap/detox-cache"

	"github.com/magiconair/properties"
	"github.com/pingcap/go-ycsb/pkg/ycsb"
)

type pebbleDetoxDB struct {
	detox *detoxcache.DeToXCache

	keyMutex   sync.Mutex
	totalKeys  uint64
	uniqueKeys map[string]struct{}
}

type pebbleDetoxCreator struct{}

func init() {
	ycsb.RegisterDBCreator("pebbledetox", &pebbleDetoxCreator{})
}

// Create implements ycsb.DBCreator.
func (c *pebbleDetoxCreator) Create(p *properties.Properties) (ycsb.DB, error) {
	fmt.Println(">>> registering pebble DeToX wrapper... <<<")
	capacity := p.GetInt("pebble.detox_capacity", 1024)
	detox := detoxcache.NewDeToXCache(capacity)
	return &pebbleDetoxDB{
		detox:      detox,
		uniqueKeys: make(map[string]struct{}),
	}, nil
}

func (db *pebbleDetoxDB) recordKey(k []byte) {
	db.keyMutex.Lock()
	db.totalKeys++
	db.uniqueKeys[string(k)] = struct{}{}
	db.keyMutex.Unlock()
}

// CleanupThread implements ycsb.DB.
func (db *pebbleDetoxDB) CleanupThread(ctx context.Context) {}

// Close implements ycsb.DB.
func (db *pebbleDetoxDB) Close() error {
	stats := db.detox.Stats()

	hits := float64(stats.Hits)
	misses := float64(stats.Misses)
	totalLookups := hits + misses

	hitRate := 0.0
	if totalLookups > 0 {
		hitRate = hits / totalLookups * 100.0
	}

	fmt.Printf("=== DeToX Cache Stats ===\n")
	fmt.Printf("  Hits:       %d\n", stats.Hits)
	fmt.Printf("  Misses:     %d\n", stats.Misses)
	fmt.Printf("  Hit Rate:   %.2f%%\n", hitRate)
	fmt.Printf("  Admissions: %d\n", stats.Admissions)
	fmt.Printf("  Evictions:  %d\n", stats.Evictions)
	fmt.Printf("  Prefetches:  %d\n", stats.Prefetches)
	fmt.Printf("  Transactions:  %d\n", stats.Transactions)
	fmt.Printf("=======================\n")


	db.keyMutex.Lock()
	total := db.totalKeys
	unique := uint64(len(db.uniqueKeys))
	db.keyMutex.Unlock()
	repeated := total - unique

	fmt.Printf("=== Key Reuse Stats ===\n")
	fmt.Printf("  Total key accesses: %d\n", total)
	fmt.Printf("  Unique keys:        %d\n", unique)
	fmt.Printf("  Repeated accesses:  %d\n", repeated)
	if total > 0 {
		fmt.Printf("  Repeat ratio:       %.2f%%\n", float64(repeated)/float64(total)*100.0)
	}
	fmt.Printf("=======================\n")

	return nil
}

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

// Delete implements ycsb.DB.
func (db *pebbleDetoxDB) Delete(ctx context.Context, table string, key string) error {
	return nil
}

// InitThread implements ycsb.DB.
func (db *pebbleDetoxDB) InitThread(ctx context.Context, threadID int, threadCount int) context.Context {
	return ctx
}

// Insert implements ycsb.DB.
func (db *pebbleDetoxDB) Insert(ctx context.Context, table string, key string, values map[string][]byte) error {
	k := encodeKey(table, key)
	data := encodeValues(values)
	db.recordKey(k)

	_, err := db.detox.Set(k, data, true, true)
	return err
}

// Read implements ycsb.DB.
func (db *pebbleDetoxDB) Read(ctx context.Context, table string, key string, fields []string) (map[string][]byte, error) {
	k := encodeKey(table, key)

	db.recordKey(k)

	val, found, err := db.detox.Get(k, true)
	if err != nil {
		return nil, err
	}
	if !found {
		// YCSB convention: not found -> (nil, nil)
		return nil, nil
	}

	data := append([]byte(nil), val...)
	return decodeValues(data, fields), nil
}

// Scan implements ycsb.DB.
func (db *pebbleDetoxDB) Scan(ctx context.Context, table string, startKey string, count int, fields []string) ([]map[string][]byte, error) {
	return nil, nil
}

// Update implements ycsb.DB.
func (db *pebbleDetoxDB) Update(ctx context.Context, table string, key string, values map[string][]byte) error {
	return db.Insert(ctx, table, key, values)
}
