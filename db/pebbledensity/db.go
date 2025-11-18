package pebbledensity

import (
	"bytes"
	"context"
	"fmt"
	"sync"

	densitycache "github.com/rohannunu/pebble-cs598rap/density-cache"

	"github.com/magiconair/properties"
	"github.com/pingcap/go-ycsb/pkg/ycsb"
)

type pebbleDensityDB struct {
	density *densitycache.DensityCache

	keyMutex   sync.Mutex
	totalKeys  uint64
	uniqueKeys map[string]struct{}
}

type pebbleDensityCreator struct{}

func init() {
	ycsb.RegisterDBCreator("pebbledensity", &pebbleDensityCreator{})
}

func (c *pebbleDensityCreator) Create(p *properties.Properties) (ycsb.DB, error) {
	fmt.Println(">>> registering pebble DensityCache wrapper... <<<")
	capacity := p.GetInt("pebble.density_capacity", 1024)
	density := densitycache.NewDensityCache(capacity)
	return &pebbleDensityDB{
		density:        density,
		uniqueKeys: make(map[string]struct{}),
	}, nil
}

func (db *pebbleDensityDB) recordKey(k []byte) {
	db.keyMutex.Lock()
	db.totalKeys++
	db.uniqueKeys[string(k)] = struct{}{}
	db.keyMutex.Unlock()
}

// Close implements ycsb.DB.
func (db *pebbleDensityDB) Close() error {
	stats := db.density.Stats()

	hits := float64(stats.Hits)
	misses := float64(stats.Misses)
	totalLookups := hits + misses

	hitRate := 0.0
	if totalLookups > 0 {
		hitRate = hits / totalLookups * 100.0
	}

	fmt.Printf("=== Density Cache Stats ===\n")
	fmt.Printf("  Hits:       %d\n", stats.Hits)
	fmt.Printf("  Misses:     %d\n", stats.Misses)
	fmt.Printf("  Hit Rate:   %.2f%%\n", hitRate)
	fmt.Printf("  Admissions: %d\n", stats.Admissions)
	fmt.Printf("  Evictions:  %d\n", stats.Evictions)
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

// InitThread implements ycsb.DB.
func (db *pebbleDensityDB) InitThread(ctx context.Context, threadID int, threadCount int) context.Context {
	return ctx
}

// CleanupThread implements ycsb.DB.
func (db *pebbleDensityDB) CleanupThread(ctx context.Context) {}


// Delete implements ycsb.DB.
func (db *pebbleDensityDB) Delete(ctx context.Context, table string, key string) error {
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

// Insert implements ycsb.DB.
func (db *pebbleDensityDB) Insert(ctx context.Context, table string, key string, values map[string][]byte) error {
	k := encodeKey(table, key)
	data := encodeValues(values)
	db.recordKey(k)

	_, err := db.density.Set(k, data, true)
	return err
}

// Read implements ycsb.DB.
func (db *pebbleDensityDB) Read(ctx context.Context, table string, key string, fields []string) (map[string][]byte, error) {
	k := encodeKey(table, key)

	db.recordKey(k)

	val, found, err := db.density.Get(k)
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
func (p *pebbleDensityDB) Scan(ctx context.Context, table string, startKey string, count int, fields []string) ([]map[string][]byte, error) {
	return nil, nil
}

// Update implements ycsb.DB.
func (db *pebbleDensityDB) Update(ctx context.Context, table string, key string, values map[string][]byte) error {
	return db.Insert(ctx, table, key, values)
}
