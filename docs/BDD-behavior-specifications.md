# Behavior Driven Development (BDD) Specifications
## EEGPT Clinical Decision Support System

### Document Control
- **Version**: 1.0.0
- **Status**: Draft
- **Last Updated**: 2025-01-17
- **Format**: Gherkin
- **Test Framework**: pytest-bdd

### Overview
This document contains behavior specifications written in Gherkin format for automated testing and validation of system behavior. Each feature corresponds to a functional requirement from the PRD.

---

## Feature: EEG File Upload and Validation

```gherkin
Feature: EEG File Upload and Validation
  As an EEG technologist
  I want to upload EEG files for analysis
  So that I can get automated quality control and clinical insights

  Background:
    Given I am authenticated as an EEG technologist
    And the system is ready to accept uploads

  Scenario: Successful EDF file upload
    Given I have a valid EDF file "normal_eeg_20min.edf"
    When I upload the file through the web interface
    Then I should receive a job ID
    And the upload status should be "processing"
    And the file should be stored securely in S3

  Scenario: Invalid file format rejection
    Given I have an invalid file "document.pdf"
    When I attempt to upload the file
    Then I should receive an error "Invalid file format. Only EDF/BDF files accepted"
    And no job should be created

  Scenario: Oversized file handling
    Given I have an EDF file larger than 1GB
    When I upload the file
    Then the system should chunk the upload
    And show upload progress percentage
    And complete successfully within 5 minutes

  Scenario Outline: Multiple file format support
    Given I have a file in "<format>" format
    When I upload the file
    Then the system should <result>

    Examples:
      | format | result                    |
      | EDF    | accept and process        |
      | BDF    | accept and process        |
      | CSV    | reject with error message |
      | FIF    | reject with error message |
```

---

## Feature: Quality Control Analysis

```gherkin
Feature: Quality Control Analysis
  As a neurologist
  I want automated quality control of EEG recordings
  So that I can identify technical issues before clinical review

  Background:
    Given an EEG file has been uploaded
    And preprocessing is complete

  Scenario: Bad channel detection
    Given the EEG has channels ["Fp1", "Fp2", "T3", "T4"]
    And channels ["T3", "T4"] have impedance > 50kΩ
    When quality control analysis runs
    Then the bad channels report should list ["T3", "T4"]
    And the bad channel percentage should be 50%
    And the QC flag should be "needs_review"

  Scenario: Clean recording validation
    Given all channels have impedance < 10kΩ
    And artifact percentage is < 5%
    When quality control analysis runs
    Then the QC report should show "good_quality"
    And no bad channels should be reported
    And the recording should proceed to clinical analysis

  Scenario: Artifact threshold detection
    Given the recording contains muscle artifacts
    When the artifact detection runs
    Then artifacts should be marked at timestamps:
      | start_time | end_time | type   |
      | 00:01:23   | 00:01:45 | muscle |
      | 00:05:12   | 00:05:18 | muscle |
    And the artifact percentage should be calculated
```

---

## Feature: Abnormality Detection

```gherkin
Feature: Abnormality Detection
  As an epileptologist
  I want automated abnormality detection
  So that abnormal EEGs are prioritized for review

  Scenario: Normal EEG classification
    Given an EEG recording with normal background rhythm
    And no epileptiform discharges present
    When the abnormality detection model runs
    Then the classification should be "normal"
    And the confidence score should be > 0.8
    And the triage flag should be "routine"

  Scenario: Abnormal EEG flagging
    Given an EEG recording with epileptiform discharges
    When the abnormality detection model runs
    Then the classification should be "abnormal"
    And the confidence score should be > 0.7
    And the triage flag should be "expedite"
    And an alert should be sent to the reviewing neurologist

  Scenario: Uncertainty handling
    Given an EEG with ambiguous patterns
    When the model confidence is between 0.4 and 0.6
    Then the classification should be "uncertain"
    And the triage flag should be "expert_review"
    And both normal and abnormal probabilities should be reported
```

---

## Feature: Epileptiform Event Detection

```gherkin
Feature: Epileptiform Event Detection
  As an EEG reviewer
  I want automated detection of epileptiform events
  So that I can quickly navigate to clinically significant segments

  Scenario: Spike detection
    Given an EEG containing interictal spikes
    When event detection runs
    Then spikes should be detected with properties:
      | timestamp | channels    | confidence |
      | 00:07:23  | ["F3","C3"] | 0.92      |
      | 00:12:45  | ["T3","T5"] | 0.87      |
    And events should be ranked by clinical significance

  Scenario: GPED pattern recognition
    Given an EEG with generalized periodic epileptiform discharges
    When pattern detection runs
    Then GPED should be identified
    And the pattern frequency should be calculated as "2 Hz"
    And duration should be marked as "00:15:00 to 00:18:30"

  Scenario: False positive minimization
    Given an EEG with eye blink artifacts
    When event detection runs
    Then eye movements should be classified as "EYEM"
    And should NOT be marked as epileptiform
    And confidence for artifact should be > 0.9
```

---

## Feature: Sleep Stage Classification

```gherkin
Feature: Sleep Stage Classification
  As a sleep specialist
  I want automated sleep staging
  So that I can assess sleep architecture in routine EEGs

  Background:
    Given YASA sleep staging model is loaded
    And the EEG duration is > 20 minutes

  Scenario: Full sleep cycle detection
    Given an EEG recording during sleep
    When sleep analysis runs
    Then sleep stages should be classified as:
      | epoch | stage | confidence |
      | 1     | Wake  | 0.95       |
      | 10    | N1    | 0.82       |
      | 25    | N2    | 0.91       |
      | 45    | N3    | 0.88       |
      | 60    | REM   | 0.93       |

  Scenario: Sleep metrics calculation
    Given sleep stages have been classified
    When metrics are calculated
    Then the report should include:
      | metric           | value |
      | Total Sleep Time | 35min |
      | Sleep Efficiency | 87.5% |
      | REM %           | 22%   |
      | N3 %            | 18%   |
      | WASO            | 5min  |

  Scenario: Wake-only recording
    Given an EEG with no sleep stages
    When sleep analysis runs
    Then all epochs should be classified as "Wake"
    And sleep metrics should show "No sleep detected"
```

---

## Feature: Report Generation

```gherkin
Feature: Clinical Report Generation
  As a healthcare provider
  I want comprehensive analysis reports
  So that I can make informed clinical decisions

  Scenario: PDF report generation
    Given all analyses are complete for job "12345"
    When I request a PDF report
    Then the report should include:
      | section               | content                          |
      | Patient Info         | De-identified ID and recording date |
      | QC Summary           | Channel quality and artifacts     |
      | Clinical Findings    | Abnormality flag and events      |
      | Sleep Analysis       | Hypnogram and metrics            |
      | Recommendations      | Suggested actions                |
    And the report should be generated within 10 seconds

  Scenario: JSON API response
    Given analysis is complete
    When I request results via API
    Then the response should follow schema:
      """json
      {
        "job_id": "string",
        "status": "completed",
        "qc": {
          "bad_channels": ["array"],
          "quality_score": "number"
        },
        "abnormal": {
          "classification": "string",
          "confidence": "number"
        },
        "events": ["array of objects"],
        "processing_time": "number"
      }
      """

  Scenario: HL7 message generation
    Given analysis is complete
    And HL7 integration is enabled
    When results are ready
    Then an ORU^R01 message should be generated
    And include OBX segments for each finding
    And be sent to the configured HL7 endpoint
```

---

## Feature: User Access Control

```gherkin
Feature: User Access Control
  As a system administrator
  I want role-based access control
  So that PHI is protected and users see appropriate data

  Scenario Outline: Role-based permissions
    Given I am logged in as "<role>"
    When I attempt to "<action>"
    Then the system should "<result>"

    Examples:
      | role          | action              | result  |
      | technologist  | upload EEG         | allow   |
      | technologist  | modify diagnosis   | deny    |
      | neurologist   | view all studies   | allow   |
      | neurologist   | delete study       | deny    |
      | admin         | manage users       | allow   |
      | researcher    | access PHI         | deny    |
      | researcher    | access deidentified| allow   |

  Scenario: Audit trail generation
    Given any user performs an action
    When the action completes
    Then an audit log entry should be created with:
      | field       | requirement          |
      | timestamp   | ISO 8601 format      |
      | user_id     | authenticated user   |
      | action      | specific action name |
      | resource    | affected resource    |
      | ip_address  | client IP           |
    And the log should be immutable
```

---

## Feature: Performance Requirements

```gherkin
Feature: System Performance
  As a user
  I want fast and reliable analysis
  So that clinical workflow is not impeded

  Scenario: Response time SLA
    Given 50 concurrent analysis requests
    When measuring API response times
    Then 95% of requests should complete within 100ms
    And 99% of requests should complete within 500ms
    And no requests should timeout

  Scenario: Processing time targets
    Given a 20-minute EEG file
    When full analysis is requested
    Then quality control should complete within 30 seconds
    And abnormality detection should complete within 60 seconds
    And full analysis should complete within 120 seconds
    And the user should see progress updates every 5 seconds

  Scenario: System availability
    Given the system is deployed in production
    When monitoring over a 30-day period
    Then uptime should be >= 99.5%
    And planned maintenance windows should be < 4 hours/month
    And unplanned downtime should be < 2 hours/month
```

---

## Implementation Notes

### Test Automation Structure
```python
# tests/features/test_eeg_upload.py
import pytest
from pytest_bdd import scenarios, given, when, then

scenarios('../features/eeg_upload.feature')

@given('I am authenticated as an EEG technologist')
def authenticated_user(test_client, auth_token):
    test_client.headers['Authorization'] = f'Bearer {auth_token}'

@when('I upload the file through the web interface')
def upload_file(test_client, test_file):
    response = test_client.post(
        '/api/v1/eeg/upload',
        files={'file': test_file},
        data={'metadata': {'patient_id': 'TEST001'}}
    )
    return response

@then('I should receive a job ID')
def verify_job_id(upload_response):
    assert 'job_id' in upload_response.json()
    assert len(upload_response.json()['job_id']) == 36  # UUID format
```

### Continuous Validation
- All scenarios run on every commit
- Integration with CI/CD pipeline
- Performance scenarios run nightly
- Security scenarios run weekly
- Full regression suite before releases

### Living Documentation
- Scenarios serve as executable specifications
- Automatically generated documentation from tests
- Stakeholder review of scenarios before implementation
- Regular scenario refinement based on user feedback