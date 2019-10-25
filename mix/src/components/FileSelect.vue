<template>
  <label class="file-select">
    <div class="select-button">
      <span style="padding-top:18px; display: block;">Mix<br>Face</span>
    </div>
    <input type="file" @change="handleFileChange" multiple/>
  </label>
</template>

<script>
export default {
  data() {
    return {
      files: []
    }
  },
  methods: {
    handleFileChange(e) {
      this.files = []
      let that = this
      let data = new FormData();
      for(let i = 0; i < e.target.files.length; i++) {
        let file = e.target.files[i]
        data.append("img[]", file, file.name)
        let reader = new FileReader()
        reader.readAsDataURL(file)
        reader.onload = function(f) {
          that.files.push({data: f.target.result})
        }
      }
      this.$emit('input', this.files)
      let req = new XMLHttpRequest()
      req.onreadystatechange = function() {
        if(req.readyState === 4 && req.status === 200) {
          that.$emit('success')
        }
      }
      req.open('POST', '/api/upload')
      req.send(data)
    }
  }
}
</script>

<style scoped>
@-webkit-keyframes breathe {
    0% {
        opacity: .4;
        box-shadow: 0 1px 2px rgba(0, 147, 223, 0.4), 0 1px 1px rgba(0, 147, 223, 0.1) inset;
    }

    100% {
        opacity: 1;
        border: 1px solid rgba(59, 235, 235, 0.7);
        box-shadow: 0 1px 30px #0093df, 0 1px 20px #0093df inset;
    }
}
.file-select {
    position: absolute;
    left: 50%;
    top: 60%;
    -webkit-transform: translate(-50%, -50%)
}
.file-select > .select-button {
  padding: 1rem;
  height: 80px;
  width: 80px;

  color: white;
  background-color: #2EA169;

  cursor: pointer;

  border-radius: 50%;
  border: 1px solid #2b92d4;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
  -webkit-animation-timing-function: ease-in-out;
  -webkit-animation-name: breathe;
  -webkit-animation-duration: 1500ms;
  -webkit-animation-iteration-count: infinite;
  -webkit-animation-direction: alternate;

  text-align: center;
  font-weight: bold;
}

.file-select > input[type="file"] {
  display: none;
}
</style>