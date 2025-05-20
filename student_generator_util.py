import os
import random
import csv

from faker import Faker


def generate_student_data_csv(file_path: str, num_students: int = 300, course_price: int = 250):
    """
    Generates synthetic student enrollment data and saves it to a CSV file.

    Each student is assigned a random semester (1 to 8), a selection of courses based on that semester,
    a random discount rate, and calculated tuition fees based on the selected courses and discount.

    Args:
        file_path (str): The output path for the generated CSV file.
        num_students (int): Number of students to generate.
        course_price (int): Price per course in the tuition calculation.
    Returns:
        None
    """
    if os.path.exists(file_path):
        return "Students already exist"

    faker = Faker()
    discounts = [0, 0.3, 0.5, 1]

    courses_by_semester = {
        1: ["Introduction to Informatics 1", "Introduction to Computer Architecture", "Discrete Structures",
            "Fundamentals of Programming (Exercises & Laboratory)", "English 1"],
        2: ["Introduction to Informatics 2", "Basic Principles of Operating Systems and System Software",
            "Analysis for Informatics", "Laboratory: Computer Organization and Computer Architecture", "English 2"],
        3: ["Databases 1", "Fundamentals of Algorithms and Data Structure", "Linear Algebra for Informatics",
            "Minor subject 1", "Minor subject 2"],
        4: ["Scripting Languages", "Introduction to Theory of Computation", "Discrete Probability Theory",
            "Minor subject 3", "Minor subject 4"],
        5: ["Numerical Programming", "Introduction to Software Engineering", "Minor subject 5",
            "Elective 1", "Minor subject 6"],
        6: ["Introduction to Computer Networking and Distributed Systems", "Databases 2", "Elective 2",
            "Software Engineering Practical Course (Project System Development)", "Minor subject 7"],
        7: ["Elective 3", "Elective 4", "Elective 5", "Internship"],
        8: ["Elective 6", "Elective 7", "Bachelor's Thesis/Capstone Project"]
    }

    students = []

    for _ in range(num_students):
        semester = random.randint(1, 8)
        all_courses = []
        total_courses = 0

        for sem in range(1, semester + 1):
            available_courses = courses_by_semester[sem]
            num_selected = random.randint(2, len(available_courses))
            selected_courses = random.sample(available_courses, k=num_selected)
            all_courses.extend(selected_courses)
            total_courses += num_selected

        discount_rate = random.choice(discounts)
        total_tuition = total_courses * course_price * (1 - discount_rate)

        students.append({
            "name": faker.first_name(),
            "surname": faker.last_name(),
            "nationality": faker.country(),
            "semester": semester,
            "all_courses": ", ".join(all_courses),
            "discount_rate": f"{int(discount_rate * 100)}%",
            "tuition_fees": round(total_tuition, 2)
        })

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "surname", "nationality", "semester", "all_courses",
                                                  "discount_rate", "tuition_fees"])
        writer.writeheader()
        writer.writerows(students)

    return "Students have been created"
