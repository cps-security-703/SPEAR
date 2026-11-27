# Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: Duplicate ID Error - "UNKNOWN"

**Error Message:**
```
chromadb.errors.DuplicateIDError: Expected IDs to be unique, found duplicates of: UNKNOWN in add.
```

**Cause:** Some MITRE techniques don't have proper external IDs in the STIX data.

**Solution:** ✅ **FIXED** - The MITRE collector now generates unique IDs using fallback methods:
1. First tries to get the official MITRE technique ID (e.g., T0866)
2. Falls back to STIX ID if no external ID exists
3. Last resort: uses technique name as ID

**What was changed:**
- Updated `collectors/mitre_collector.py` - `_extract_technique_id()` method
- Now generates unique IDs like `MITRE-abc12345` or `MITRE-Block-Command-Message`

---

### Issue 2: NVD API 404 Errors

**Error Message:**
```
2026-02-04 19:33:58 | ERROR    | NVD API error: 404
```

**Cause:** The NVD API might be:
1. Returning 404 for keywords with no results
2. Having endpoint changes
3. Rate limiting without proper error codes

**Solution:** ✅ **FIXED** - Updated error handling:
- 404 errors now logged as debug (not error) when no results found
- Better error messages showing response text
- Added warning summary if no CVEs collected

**Workaround:** If NVD continues to fail, skip it:
```bash
python main.py --skip-nvd
```

This will still create a database with:
- 95 MITRE ATT&CK techniques
- 17 STRIDE patterns
- 10 MITRE-STRIDE mappings
- 14 CICEVSE2024 attack patterns
- **Total: ~136 documents**

---

### Issue 3: NVD API Rate Limiting

**Symptoms:**
- Slow CVE collection
- 403 errors
- "Rate limit exceeded" warnings

**Solutions:**

1. **Use API Key** (Recommended):
   - Get free API key from: https://nvd.nist.gov/developers/request-an-api-key
   - Add to `.env`: `NVD_API_KEY=your_key_here`
   - Increases rate limit from 5 req/30s to 50 req/30s

2. **Reduce CVE Count**:
   ```bash
   python main.py --nvd-max-results 20
   ```

3. **Skip NVD Temporarily**:
   ```bash
   python main.py --skip-nvd
   ```

---

### Issue 4: ChromaDB Permission Errors

**Error Message:**
```
PermissionError: [WinError 32] The process cannot access the file
```

**Cause:** ChromaDB files locked by another process.

**Solution:**
1. Close all Python processes
2. Delete ChromaDB directory:
   ```bash
   rmdir /s chroma_db  # Windows
   rm -rf chroma_db    # Linux/Mac
   ```
3. Re-run pipeline

---

### Issue 5: Embedding Model Download Fails

**Error Message:**
```
Failed to load embedding model: ...
```

**Cause:** Network issues or missing dependencies.

**Solutions:**

1. **Check Internet Connection**

2. **Manually Download Model**:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   ```

3. **Use Different Model**:
   Edit `config.py`:
   ```python
   EMBEDDING_MODEL = "all-mpnet-base-v2"  # Larger, more accurate
   # or
   EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"  # Smaller, faster
   ```

---

### Issue 6: Gemini API Errors

**Error Message:**
```
Error generating analysis: ...
```

**Causes & Solutions:**

1. **No API Key Set**:
   ```bash
   # Add to .env
   GOOGLE_API_KEY=your_key_here
   ```

2. **Invalid API Key**:
   - Get new key from: https://makersuite.google.com/app/apikey
   - Verify it's copied correctly (no extra spaces)

3. **Rate Limiting**:
   - Wait a few minutes
   - Reduce query frequency

4. **Content Policy Violation**:
   - Rephrase your query
   - Avoid overly aggressive attack scenarios

---

### Issue 7: CICEVSE2024 Dataset Not Found

**Error Message:**
```
Failed to load dataset: [Errno 2] No such file or directory
```

**Solution:**
1. Verify file exists: `data/CICEVSE2024_NT.csv`
2. Use absolute path:
   ```bash
   python main.py --cicevse-dataset "C:/full/path/to/CICEVSE2024_NT.csv"
   ```
3. Skip if not available:
   ```bash
   python main.py --skip-cicevse
   ```

---

### Issue 8: Memory Errors with Large Datasets

**Error Message:**
```
MemoryError: Unable to allocate array
```

**Cause:** CICEVSE2024 has 1.2M+ records.

**Solutions:**

1. **Sample Dataset**:
   ```python
   # In cicevse_collector.py, modify load_dataset:
   df = pd.read_csv(filepath, nrows=100000)  # Load only 100k rows
   ```

2. **Increase System Memory**

3. **Process in Batches**:
   - Modify collector to process dataset in chunks

---

### Issue 9: Slow Embedding Generation

**Symptoms:**
- Embedding phase takes very long
- High CPU usage

**Solutions:**

1. **Use GPU** (if available):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Reduce Batch Size**:
   Edit `pipeline.py`:
   ```python
   embeddings = self.embedder.embed_batch(texts, batch_size=16)  # Default: 32
   ```

3. **Use Smaller Model**:
   Edit `config.py`:
   ```python
   EMBEDDING_MODEL = "all-MiniLM-L3-v2"  # Faster, smaller
   ```

---

### Issue 10: Database Query Returns No Results

**Symptoms:**
- `query_db.py` returns 0 results
- Database appears empty

**Diagnosis:**
```bash
# Check database stats
python -c "from vector_db import ChromaDBManager; db = ChromaDBManager(); print(db.get_collection_stats())"
```

**Solutions:**

1. **Database Not Created**:
   ```bash
   python main.py --skip-nvd  # Quick creation
   ```

2. **Wrong Collection Name**:
   - Check `config.py` - `COLLECTION_NAME`
   - Verify `.env` settings

3. **Query Too Specific**:
   - Try broader queries
   - Remove filters

---

## Debugging Commands

### Check Database Status
```bash
python -c "from vector_db import ChromaDBManager; db = ChromaDBManager(); print(db.get_collection_stats())"
```

### Test Embedding
```bash
python -c "from vector_db import DocumentEmbedder; e = DocumentEmbedder(); print(len(e.embed_text('test')))"
```

### Test NVD API
```bash
python -c "from collectors import NVDCollector; c = NVDCollector(); print(c._search_by_keyword('SCADA', '2024-01-01'))"
```

### Verify Gemini API
```bash
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); m = genai.GenerativeModel('gemini-pro'); print(m.generate_content('test').text)"
```

---

## Getting Help

### Enable Debug Logging

Edit `main.py`:
```python
logger.add(
    sys.stderr,
    level="DEBUG"  # Change from INFO to DEBUG
)
```

### Check Log Files

```bash
# View pipeline log
cat pipeline.log  # Linux/Mac
type pipeline.log  # Windows
```

### Report Issues

When reporting issues, include:
1. Full error message and traceback
2. Command you ran
3. Python version: `python --version`
4. OS and version
5. Contents of `.env` (without API keys)
6. Output of database stats command

---

## Prevention Tips

1. **Always Test First**:
   ```bash
   python main.py --skip-nvd --nvd-max-results 5
   ```

2. **Use Version Control**:
   - Commit working configurations
   - Track changes to collectors

3. **Monitor Resources**:
   - Check disk space before running
   - Monitor memory usage with large datasets

4. **Backup Database**:
   ```bash
   python main.py --export-db backup.json
   ```

5. **Keep Dependencies Updated**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Duplicate IDs | ✅ Fixed in code - re-run pipeline |
| NVD 404 errors | Use `--skip-nvd` flag |
| Rate limiting | Add NVD API key to `.env` |
| No Gemini key | Add `GOOGLE_API_KEY` to `.env` |
| Database locked | Delete `chroma_db/` folder |
| Out of memory | Reduce dataset size or batch size |
| Slow embedding | Use smaller model or GPU |
| No results | Verify database created successfully |

---

## Still Having Issues?

1. Check `README.md` for detailed documentation
2. Review `QUICKSTART.md` for setup steps
3. Try the example scripts in `examples/`
4. Re-run with `--skip-nvd` to isolate NVD issues
