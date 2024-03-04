import pandas as pd
import Levenshtein  # Importing Levenshtein for string similarity

class HackathonScorer:
    def __init__(self, applications_path='./data/applications/applications.csv', num_applications = -1, scorer_initials = 'XX'):
        self.applications_path = applications_path
        self.num_applications = num_applications
        self.scorer_initials = scorer_initials

    def load_applications(self):
        applications = pd.read_csv(self.applications_path)
        if self.num_applications > 0:
            applications = applications.sample(self.num_applications)
        # Drop unnecessary columns
        applications = applications.drop(columns=['email', 'guest_type', 'rsvp_status', 'registration_time', 'create_time',
                                                  'update_time', 'invite_code', 'registration_code', 'region_code',
                                                  'marketing_opt_in', 'subscribers', 'source', 'consentphotos',
                                                  'company', 'diet'])
        # Add an ID column
        applications['id'] = range(1, len(applications) + 1)
        # Rename columns for clarity
        applications = applications.rename(columns={'whichcloud': 'Which cloud platforms have you used?',
                                                    'dataanalysis': 'How confident are you in performing data analysis?',
                                                    'startup': 'Have you worked at a startup before?',
                                                    'education': 'What is your current level of education?',
                                                    'fieldstudy': 'What field of study are you in?',
                                                    'full_event': 'Are you able to attend the full event?',
                                                    'ml': 'How confident are you in machine learning?',
                                                    'python': 'How confident are you in Python?',
                                                    'whyjoin': 'Why do you want to join the hackathon?',
                                                    'cloudplatform': 'How confident are you in using cloud platforms?'})
        # Convert the 'isgroup' column to boolean
        applications['isgroup'] = applications['isgroup'].apply(lambda x: True if x == 'Group (Max 4 people)' else False)
        return applications

    def save_applications(self, applications, path='./data/applications/applications_scored.csv'):
        applications.to_csv(path, index=False)

    def id_groups(self, applications):
        groups = {}  # Dictionary to store groups and their members
        individual_applicants = []  # List to store individual applicants

        # Iterate through each application
        for index, row in applications.iterrows():
            group_names = row['groupnames']
            if pd.notna(group_names):  # Check if groupnames field is not empty
                names = [name.strip() for name in group_names.split(',')]
                if row['isgroup']:
                    group_id = row['id']
                    groups[group_id] = names
                else:
                    for name in names:
                        found_group = False
                        for group_id, group_members in groups.items():
                            if name in group_members:
                                found_group = True
                                break
                        if not found_group:
                            individual_applicants.append(name)

        # Merge similar groups and trim group members
        return self.trim_groups(groups)

    def trim_groups(self, groups):
        trimmed_groups = {}

        for group_id, group_members in groups.items():
            sorted_members = sorted(group_members)
            merged_names = {}

            for member in sorted_members:
                for other_group_id, other_group_members in groups.items():
                    if other_group_id != group_id:
                        for other_member in other_group_members:
                            if self._are_similar(member, other_member):
                                if len(member) >= len(other_member):
                                    merged_names[member] = member
                                else:
                                    merged_names[member] = other_member

            for member in sorted_members:
                if member not in merged_names:
                    merged_names[member] = member

            trimmed_groups[group_id] = list(merged_names.values())

        return trimmed_groups

    def _are_similar(self, name1, name2):
        threshold = 3
        return Levenshtein.distance(name1.lower(), name2.lower()) < threshold

    def score_applications(self, applications):
        scores = []
        exclude_field_list = ['id', 'first_name', 'last_name', 'uni', 'isgroup', 'group', 'groupnames', 'score', 'type']
        for index, application in applications.iterrows():
            score = 0
            maxscore = (len(application) - len(exclude_field_list) + 2) * 5
            print("--------------------")
            print(f"Application {application['id']}\nApplicant {application['first_name']} {application['last_name']} ({application['type']}) from {application['uni']}")
            if application['isgroup']:
                print(f"This is a group application with group members: {application['groupnames']}")
            print("--------------------")
            print(f"Scoring application {application['id']}")
            for field in application.index:
                if field in exclude_field_list:
                    continue
                response = application[field]
                print(f"{field}: {response}")
                pointscore = input(f"Score for (0-5): ")
                if pointscore == '':
                    pointscore = 0
                score += int(pointscore)
            scores.append(score)
            print(f"Total score for application {application['id']}: {score}/{maxscore}\n")
        applications[scorer_initials] = scores
        return applications

    def group_applications(self, applications):
        groups = applications.groupby('group')
        group_scores = groups[scorer_initials].mean()
        return group_scores

    def main(self):
        applications = self.load_applications()
        groups = self.id_groups(applications)
        applications = self.score_applications(applications)
        self.save_applications(applications)
        # group_scores = self.group_applications(applications)
        # print(group_scores)

if __name__ == "__main__":
    scorer_initials = input("\nWelcome!\n\nEnter your initials: ")
    scorer_initials += "_score"
    num_applications = int(input("\nEnter the number of applications to score (-1 for all contestants): "))
    hackathon_scorer = HackathonScorer(applications_path='./data/applications/applications.csv', num_applications = num_applications, scorer_initials = scorer_initials)
    hackathon_scorer.main()
