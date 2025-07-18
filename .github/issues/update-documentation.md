# Update Documentation for Recent Changes

## Current Status
We've made significant changes that need to be documented:
- Added Redis caching with auto-reconnect
- Integrated EDFStreamer for large files
- Added authentication for cache endpoints
- Implemented utc_now() utility
- Enhanced API with health endpoint version info

## Documentation Tasks
- [ ] Update README.md with Redis setup instructions
- [ ] Document EDF streaming thresholds and configuration
- [ ] Add API authentication examples
- [ ] Update CLAUDE.md with recent architectural decisions
- [ ] Create deployment guide with environment variables
- [ ] Add troubleshooting section for common issues
- [ ] Document performance tuning parameters

## Specific Sections Needed
1. **Redis Configuration**
   - Environment variables (REDIS_URL, REDIS_PASSWORD)
   - Auto-reconnect behavior
   - Cache key patterns
   - TTL settings

2. **Authentication**
   - Admin token generation
   - HMAC signature creation
   - Protected endpoint usage

3. **Streaming Configuration**
   - Memory thresholds
   - Window sizes and overlaps
   - Performance implications

4. **API Usage Examples**
   - cURL examples for all endpoints
   - Python client examples
   - Error handling patterns

## Priority
Medium - Important for onboarding and maintenance

@claude Please update the project documentation to reflect all recent changes, including Redis caching, EDF streaming, authentication, and API enhancements. Focus on practical examples and deployment guidance.
