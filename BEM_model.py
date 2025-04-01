import mne

# Define the subjects directory where fsaverage will be downloaded
subjects_dir = mne.get_config("SUBJECTS_DIR")  # Check if it's set
if subjects_dir is None:
    subjects_dir = "C:/Users/thorb/PycharmProjects/Internship II - Psychiatric Analysis/preprocessed_data"
    mne.set_config("SUBJECTS_DIR", subjects_dir)

# Download fsaverage if not already available
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
print(f"fsaverage downloaded to {subjects_dir}")

subjects_dir = "preprocessed_data"
subject = "fsaverage"  # Use a standard brain or your own subject

# Step 1: Create BEM model (if missing)
bem_model = mne.make_bem_model(subject=subject, subjects_dir=subjects_dir)

# Step 2: Convert it into a BEM solution
bem_solution = mne.make_bem_solution(bem_model)

# Step 3: Save it
mne.write_bem_solution(f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif", bem_solution)
print(f"BEM model saved for {subject}!")