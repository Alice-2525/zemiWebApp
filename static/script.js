document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('fileInput').addEventListener('change', function(event) {
      var file = event.target.files[0];
      if (file) {
          var reader = new FileReader();
          reader.onload = function(e) {
              var imagePreview = document.getElementById('imagePreview');
              imagePreview.src = e.target.result;
              imagePreview.style.display = 'block';
          }
          reader.readAsDataURL(file);
      }
  });
});
