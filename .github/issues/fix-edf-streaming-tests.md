# Fix Remaining EDF Streaming Tests (17 failures)

## Current Status
We have 17 failing tests in the EDF streaming module that need to be fixed. The EDFStreamer class has been partially implemented but needs completion.

## Failing Tests
- `test_streamer_context_manager` - ✅ Fixed (added `__enter__` and `__exit__` methods)
- `test_stream_chunks` - Needs implementation
- `test_process_in_windows` - Needs implementation
- `test_process_in_windows_with_overlap` - Needs implementation
- `test_get_info` - ✅ Fixed (renamed from `get_file_info`)
- `test_invalid_overlap` - Needs validation logic
- `test_estimate_memory_usage` - ✅ Fixed (added import)
- `test_process_large_edf_streaming_decision` - Needs implementation
- `test_process_small_edf_no_streaming` - Needs implementation

## Tasks
- [ ] Complete implementation of `stream_chunks()` method
- [ ] Complete implementation of `process_in_windows()` method with overlap support
- [ ] Add validation for overlap parameter (must be between 0 and 1)
- [ ] Fix attribute references in tests (use `_raw`, `_sfreq`, etc.)
- [ ] Update `get_info()` return values to match test expectations
- [ ] Ensure memory estimation functions work correctly

## Priority
High - This is blocking the full streaming integration for large EEG files

@claude Please fix the remaining EDF streaming tests by completing the implementation of the EDFStreamer class methods.
