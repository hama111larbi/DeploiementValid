import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  prediction: any = null;
  loading = false;
  error = null;

  formData = {
    bmi: null,
    gender: '',
    years_experience: null,
    speciality: '',
    discharge_location: ''
  };

  constructor(private http: HttpClient) {}

  ngOnInit() {}

  onSubmit() {
    this.loading = true;
    this.error = null;
    this.prediction = null;

    this.http.post('http://localhost:5000/api/predict', this.formData)
      .subscribe(
        (response: any) => {
          this.prediction = response;
          this.loading = false;
        },
        (error) => {
          this.error = 'Une erreur est survenue lors de la prédiction.';
          this.loading = false;
          console.error('Error:', error);
        }
      );
  }

  resetForm() {
    this.formData = {
      bmi: null,
      gender: '',
      years_experience: null,
      speciality: '',
      discharge_location: ''
    };
    this.prediction = null;
    this.error = null;
  }
}
