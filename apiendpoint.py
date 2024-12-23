from flask import Flask, jsonify

app = Flask(__name__)

# Event data
event_data = {
  "event": {
    "title": "GDG Coimbatore Tech Event - December 7th, 2024",
    "location": "Coimbatore",
    "date": "Saturday, December 7th, 2024",
    "time": {
      "registration": "08:00 AM",
      "kickoff": "09:00 AM"
    },
    "agenda": {
      "morning_sessions": [
        {
          "time": "09:30 AM - 10:30 AM",
          "session": "Innate Human Strengths to Leverage the Transformation That AI Will Bring",
          "speaker": "Raja Chidambaram"
        },
        {
          "time": "10:30 AM - 11:00 AM",
          "session": "AI for a Boundless and Sustainable Future: Designing Solutions for Complex Societal Challenges",
          "speaker": "Prabakaran Chandran"
        },
        {
          "time": "11:00 AM - 11:30 AM",
          "session": "Beckn Protocol - Reimagining Digital Economies",
          "speaker": "Ravi Prakash"
        },
        {
          "time": "11:30 AM - 12:00 PM",
          "session": "Break"
        }
      ],
      "afternoon_tracks": {
        "12:00 PM - 01:30 PM": {
          "track_1": [
            {
              "time": "12:00 PM",
              "session": "From Flutter Widgets to AI Insights: Leveraging Google Gemini",
              "speaker": "Kamal Shree Soundirapandian"
            },
            {
              "time": "12:30 PM",
              "session": "Simplifying Kubernetes Operations Using Gen AI Models",
              "speaker": "Prasanna V"
            },
            {
              "time": "01:00 PM",
              "session": "AI-Powered Malware: The Evolving Threat Landscape",
              "speaker": "Shrutirupa Banerjiee"
            }
          ],
          "track_2": [
            {
              "time": "12:00 PM",
              "session": "Adoption Strategies for Kotlin Multiplatform",
              "speaker": "Shangeeth Sivan"
            },
            {
              "time": "12:30 PM",
              "session": "Edge AI Simplified: Choosing the Right NVIDIA Jetson Device for Your Application",
              "speaker": "Bhuvaneshwari Kanagaraj"
            },
            {
              "time": "01:00 PM",
              "session": "Build Your Next GenAI Application with RAG",
              "speaker": "Praveen Thirumurugan"
            }
          ],
          "வாங்க பேசலாம்_tech_track": [
            {
              "time": "12:00 PM",
              "session": "Generative AI in the Wild: From Frontier Breakthroughs to Real-World Impact",
              "speaker": "Goutham Nivass"
            },
            {
              "time": "12:30 PM",
              "session": "Software Development: How to Use New Methods in Your Work?",
              "speaker": "Prasanth"
            },
            {
              "time": "01:00 PM",
              "session": "Tech and Sustainability: Finding Harmony in the Digital Age",
              "speaker": "Anitha Chinnasamy"
            }
          ]
        },
        "01:30 PM - 03:00 PM": {
          "session": "Lunch Break"
        },
        "03:00 PM - 04:30 PM": {
          "track_1": [
            {
              "time": "03:00 PM",
              "session": "Securing Made Efficient - An Approach to Securing Hybrid Workloads on GCP",
              "speaker": "Imran Roshan"
            },
            {
              "time": "03:30 PM",
              "session": "Is WASM the Future of Web Development?",
              "speaker": "Yeswanth Rajakumar"
            },
            {
              "time": "04:00 PM",
              "session": "Unleashing the Potential of GKE for Next-Level Solutions",
              "speaker": "Abhishek Sharma"
            }
          ],
          "track_2": [
            {
              "time": "03:00 PM",
              "session": "Quality as a Mindset: Ensuring Excellence at Every Stage",
              "speaker": "S R Harinya Devi"
            },
            {
              "time": "03:30 PM",
              "session": "Angular Signals Unleashed: A Deep Dive into New Features and Best Practices",
              "speaker": "Jeevan"
            },
            {
              "time": "04:00 PM",
              "session": "Documenting AI Software Like a Pro: Techniques, Tools, and Emerging Trends",
              "speaker": "Aarthy Ramesh"
            }
          ],
          "track_3": [
            {
              "time": "03:00 PM",
              "session": "Core Web Vitals",
              "speaker": "Ramesh Selvam"
            },
            {
              "time": "03:30 PM",
              "session": "Nutpam: The Invisible Thread to Weave an Exceptional Design",
              "speaker": "Ramachandran A"
            },
            {
              "time": "04:00 PM",
              "session": "Decision Engineering with AI: Unlocking Next-Gen Content Intelligence for Smarter Decisions",
              "speaker": "Prabakaran Chandran"
            }
          ]
        }
      }
    },
    "concluding_sessions": {
      "time": "04:30 PM - 05:00 PM",
      "session": "Tea and Networking Session"
    },
    "wrap_up": {
      "time": "05:00 PM",
      "session": "Wrap-Up"
    },
    "contact": {
      "email": "gdgcoimbatore@example.com",
      "socials": ["Instagram", "LinkedIn", "Facebook"]
    }
  }
}

@app.route('/api/event', methods=['GET'])
def get_event_details():
    return jsonify(event_data)

if __name__ == '__main__':
    app.run(debug=True)
