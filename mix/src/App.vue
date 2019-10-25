<template>
  <div id="app">
    <vue-particles
        color="#dedede"
        :particleOpacity="0.7"
        :particlesNumber="60"
        shapeType="circle"
        :particleSize="4"
        linesColor="#fff"
        :linesWidth="1"
        :lineLinked="true"
        :lineOpacity="0.4"
        :linesDistance="150"
        :moveSpeed="2"
        :hoverEffect="true"
        hoverMode="grab"
        :clickEffect="true"
        clickMode="push"
      >
      </vue-particles>
      <div class="img-box">
        <div class="animated infinite pulse img j-img" v-for="(item, i) in files" :key="i">
          <img v-bind:src="item.data" v-bind:style="item.style"/>
        </div>
      </div>
      <FileSelect @input=uploaded @success=success></FileSelect>
  </div>
</template>

<script>
import FileSelect from './components/FileSelect.vue'

export default {
  name: 'app',
  components: {
    FileSelect
  },
  data() {
    return {
      files: [],
      inter: ''
    }
  },
  methods: {
    uploaded: function(data) {
      this.files = data

      setTimeout(function() {
        let divs = document.querySelectorAll('.j-img')
        for(let i = 0; i < divs.length; i++) {
            console.log(i)
        }
      }, 800)

      // upload
      let div = document.querySelector('.select-button')
      let i = 0
      this.inter = setInterval(function() {
        i++
        if (i >= 360) {
         i = 3
        }
        div.style.transform = 'rotate(' + i + 'deg)'
      }, 10)
    },
    success: function() {
      clearInterval(this.inter)
      let div = document.querySelector('.select-button')
      div.style.transform = 'rotate(0deg)'
      this.files = [{"data": '/img/merged_image.jpg', 'style': 'width: 400px'}]
    }
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  background-image: url('~@/assets/bg.jpg');
  background-repeat: no-repeat;
  background-size: cover;
}
.img-box {
  position: absolute;
  top: 1%;
  height: 40%;
  width: 100%;
  #background-color: red;
  display: flex;
  flex-wrap: nowrap;
  overflow: hidden;
}

.img {
  padding: 10px;
  margin: auto;
  flex-grow: 0;
  display: inline-block;
  overflow: hidden;
}

.img-box > div > img {
  width: 100%;
}
</style>
