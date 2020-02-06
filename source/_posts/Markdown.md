---
title: Markdown语法
date: 2020-02-05 18:44:24
tags: [introduce,language]
category: introduce
---

# #概念
为什么会有Markdown语言，Markdown就是简化阉割过的HTML，优点是语法简单高效，缺点就是HTML中一些稍微高级复杂一点的效果，比如文本居中，Markdown就无法实现，所以Markdown一般被用来写对页面排版要求不高，以文字为主的笔记和文档

## 什么是Markdown
Markdown 是一种轻量级标记语言，它允许人们使用易读易写的纯文本格式编写文档。
Markdown 语言在 2004 由约翰·格鲁伯（英语：John Gruber）创建。
Markdown 编写的文档可以导出 HTML 、Word、图像、PDF、Epub 等多种格式的文档。
Markdown 编写的文档后缀为 .md, .markdown。

## Markdown用在何处
Markdown 能被使用来撰写电子书，如：Gitbook。
当前许多网站都广泛使用 Markdown 来撰写帮助文档或是用于论坛上发表消息。例如：GitHub、简书、reddit、Diaspora、Stack Exchange、OpenStreetMap 、SourceForge等。【**用于写其他博客很有用，而且开源的项目多数使用Markdown进行介绍**】

常使用的工具是[Typora 编辑器](https://typora.io/)，Typora 支持 MacOS 、Windows、Linux 平台，且包含多种主题，编辑后直接渲染出效果。支持导出HTML、PDF、Word、图片等多种类型文件。

## 如何解析
···

# #语法

## 标题

标题有两种格式：

1、使用=和-标记一级和二级标题，需另起一行，不建议使用

2、使用#号标记，#号数量对应1-6级标题，只到6级

## 段落

1、段落标识无特殊格式，直接换行就行

2、字体

```
*斜体文本*
_斜体文本_
**粗体文本**
__粗体文本__
***粗斜体文本***
___粗斜体文本___
```

3、分隔线

你可以在一行中用三个以上的星号、减号、底线来建立一个分隔线，行内不能有其他东西。你也可以在星号或是减号中间插入空格。下面每种写法都可以建立分隔线：

```
***

* * *

*****

- - -

----------
```

4、删除线

段落上的文字要添加删除线，只需要在文字的两端加上两个波浪线 **~~** 即可
~~删除线~~

5、下划线

下划线可以通过 HTML 的 `<u>`标签来实现：

```
<u>带下划线文本</u>
```

6、脚注

脚注是对文本的补充说明。

Markdown 脚注的格式如下:

```
[^要注明的文本]:文本
```
创建[^BOUND]
显示[^BOUND]:菜鸟教程 -- 学的不仅是技术，更是梦想！！！

## 列表

Markdown 支持有序列表和无序列表。

1. 无序列表使用星号(*****)、加号(**+**)或是减号(**-**)作为列表标记：

```
* 第一项
* 第二项
* 第三项

+ 第一项
+ 第二项
+ 第三项


- 第一项
- 第二项
- 第三项
```

2. 有序列表使用数字并加上 **.** 号来表示，如：

```
1. 第一项
2. 第二项
3. 第三项
```

3. 列表嵌套：列表嵌套只需在子列表中的选项添加四个空格即可

## 区块

Markdown 区块引用是在段落开头使用 **>** 符号 ，然后后面紧跟一个**空格**符号：

```
> 区块引用
> 菜鸟教程
> 学的不仅是技术更是梦想
```

其实就是前面有竖线来分块，以达到突出强调

另外区块是可以嵌套的，一个 **>** 符号是最外层，两个 **>** 符号是第一层嵌套，以此类推：

```
> 最外层
> > 第一层嵌套
> > > 第二层嵌套
```

当然区块跟列表是可以嵌套使用的

## 代码

1. 如果是段落上的一个函数或片段的代码可以用反引号把它包起来（**`**），例如：

   ```
   `printf()` 函数
   ```

2. 代码区块使用 **4 个空格**或者一个**制表符（Tab 键）**。

3. 用 **```** 包裹一段代码，并指定一种语言（也可以不指定）：

   ​```javascript
   $(document).ready(function () {
       alert('RUNOOB');
   });
   ​```

   

## 链接

1. 链接使用方法如下：

   ```
   [链接名称](链接地址)
   
   或者
   
   <链接地址>
   ```

   例如：

   ```
   这是一个链接 [菜鸟教程](https://www.runoob.com)
   ```

2. 高级链接，将链接当成变量

   ```
   链接也可以用变量来代替，文档末尾附带变量地址：
   这个链接用 1 作为网址变量 [Google][1]
   这个链接用 runoob 作为网址变量 [Runoob][runoob]
   然后在文档的结尾为变量赋值（网址）
   
     [1]: http://www.google.com/
     [runoob]: http://www.runoob.com/
   ```

## 图片

1. Markdown 图片语法格式如下：

```
![alt 属性文本](图片地址)

![alt 属性文本](图片地址 "可选标题")
```

- 开头一个感叹号 !
- 接着一个方括号，里面放上图片的替代文字
- 接着一个普通括号，里面放上图片的网址，最后还可以用引号包住并加上选择性的 'title' 属性的文字。

使用实例：

```
![RUNOOB 图标](http://static.runoob.com/images/runoob-logo.png)

![RUNOOB 图标](http://static.runoob.com/images/runoob-logo.png "RUNOOB")
```

如果是当前项目的图片位置，以hexo为例：

```
![](/images/horovod/3.png "tensorflow分布式在GPU数量增加时的表现")
```
2. Markdown 还没有办法指定图片的高度与宽度，如果你需要的话，你可以使用普通的 <img> 标签。
<img src="/images/horovod/3.png" width="50%">
```
<img src="/images/horovod/3.png" width="50%">
```

## 表格

1. Markdown 制作表格使用 **|** 来分隔不同的单元格，使用 **-** 来分隔表头和其他行。

   语法格式如下：

   ```
   |  表头   | 表头  |
   |  ----  | ----  |
   | 单元格  | 单元格 |
   | 单元格  | 单元格 |
   ```

2. 对齐方式

   - **-:** 设置内容和标题栏居右对齐。
   - **:-** 设置内容和标题栏居左对齐。
   - **:-:** 设置内容和标题栏居中对齐。

   实例如下：

   ```
   | 左对齐 | 右对齐 | 居中对齐 |
   | :-----| ----: | :----: |
   | 单元格 | 单元格 | 单元格 |
   | 单元格 | 单元格 | 单元格 |
   ```

## Something

1. 支持的 HTML 元素

   不在 Markdown 涵盖范围之内的标签，都可以直接在文档里面用 HTML 撰写。

   目前支持的 HTML 元素有：`<kbd> <b> <i> <em> <sup> <sub> <br>`等 

2. Markdown 使用了很多特殊符号来表示特定的意义，如果需要显示特定的符号则需要使用转义字符，Markdown 使用反斜杠转义特殊字符：

   ```
   **文本加粗** 
   \*\* 正常显示星号 \*\*
   ```

3. 公式

   当你需要在编辑器中插入数学公式时，可以使用两个美元符 $$ 包裹 TeX 或 LaTeX 格式的数学公式来实现。提交后，问答和文章页会根据需要加载 Mathjax 对数学公式进行渲染。

4. 字体颜色如何设置，只能使用html标签吗

参考：[Markdown 教程](https://www.runoob.com/markdown/md-tutorial.html)